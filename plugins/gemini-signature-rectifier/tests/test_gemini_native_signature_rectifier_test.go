// sub2api-package: backend/internal/service
package service

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

const nativeRectifierRejected = `{"error":{"code":400,"message":"Corrupted thought signature."}}`
const nativeRectifierWrappedSignature = "EgMKAXg="
const nativeRectifierRequest = `{"contents":[{"role":"model","parts":[{"text":"reasoning","thought":true,"thoughtSignature":"EgMKAXg="},{"functionCall":{"name":"tool","args":{"value":17,"thoughtSignature":"business"}},"thoughtSignature":"EgMKAXg="}]}]}`
const nativeRectifierSuccess = `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2}}`

type nativeRectifierResponse struct {
	status int
	body   string
	err    error
}

type nativeRectifierCanceledSettingRepo struct {
	SettingRepository
	cancel context.CancelFunc
}

func (r *nativeRectifierCanceledSettingRepo) GetValue(ctx context.Context, _ string) (string, error) {
	r.cancel()
	return "", ctx.Err()
}

type nativeRectifierTrackedBody struct {
	io.Reader
	closed bool
}

func (b *nativeRectifierTrackedBody) Close() error {
	b.closed = true
	return nil
}

type nativeRectifierUpstream struct {
	HTTPUpstream
	responses      []nativeRectifierResponse
	requests       []*http.Request
	bodies         [][]byte
	responseBodies []*nativeRectifierTrackedBody
	after          func(int)
}

func (u *nativeRectifierUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	index := len(u.requests)
	u.requests = append(u.requests, req)
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	u.bodies = append(u.bodies, body)
	if u.after != nil {
		u.after(index)
	}
	if index >= len(u.responses) {
		return nil, fmt.Errorf("unexpected synthetic request %d", index)
	}
	r := u.responses[index]
	if r.err != nil {
		return nil, r.err
	}
	responseBody := &nativeRectifierTrackedBody{Reader: strings.NewReader(r.body)}
	u.responseBodies = append(u.responseBodies, responseBody)
	contentType := "application/json"
	if strings.HasPrefix(r.body, "data:") {
		contentType = "text/event-stream"
	}
	return &http.Response{
		StatusCode: r.status,
		Header:     http.Header{"Content-Type": {contentType}, "X-Request-Id": {fmt.Sprintf("synthetic-%d", index)}},
		Body:       responseBody,
	}, nil
}

func newNativeRectifierService(responses ...nativeRectifierResponse) (*GeminiMessagesCompatService, *nativeRectifierUpstream) {
	cfg := &config.Config{}
	repo := &errorPolicyRepoStub{}
	rate := NewRateLimitService(repo, nil, cfg, nil, nil)
	rate.settingService = NewSettingService(&nativeRectifierSettingRepo{value: `{"enabled":true,"gemini_native_signature_enabled":true}`}, cfg)
	upstream := &nativeRectifierUpstream{responses: responses}
	return &GeminiMessagesCompatService{
		cfg: cfg, accountRepo: repo, rateLimitService: rate, httpUpstream: upstream,
		tokenProvider: NewGeminiTokenProvider(nil, &fakeGeminiTokenCache{token: "synthetic-token"}, nil),
	}, upstream
}

func nativeRectifierContext() (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", nil)
	return c, recorder
}

func nativeRectifierKinds(c *gin.Context) []string {
	raw, _ := c.Get(OpsUpstreamErrorsKey)
	events, _ := raw.([]*OpsUpstreamErrorEvent)
	kinds := make([]string, 0, len(events))
	for _, event := range events {
		kinds = append(kinds, event.Kind)
	}
	return kinds
}

func TestNativeRectifierMatcher(t *testing.T) {
	for _, message := range []string{"Invalid thought signature.", "thoughtSignature is missing", "THOUGHT_SIGNATURE"} {
		body, err := json.Marshal(map[string]any{"error": map[string]string{"message": message}})
		require.NoError(t, err)
		require.True(t, isGeminiNativeSignatureRelatedError(body), message)
	}
	require.True(t, isGeminiNativeSignatureRelatedError([]byte(`{"error":{"message":"{\"error\":{\"message\":\"thoughtSignature\"}}"}}`)))
	for _, body := range []string{`{"error":{"message":"Invalid authentication signature"}}`, `{"echo":"thoughtSignature","error":{"message":"Bad input"}}`, `thoughtSignature`, `{}`} {
		require.False(t, isGeminiNativeSignatureRelatedError([]byte(body)), body)
	}
}

func TestNativeRectifierCleaner(t *testing.T) {
	for _, tc := range []struct {
		name, input, want string
		changed           bool
	}{
		{"thought text", `{"contents":[{"parts":[{"text":"  reason  ","thought":true,"thoughtSignature":"old","thought_signature":"old2"}]}]}`, `{"contents":[{"parts":[{"text":"  reason  "}]}]}`, true},
		{"function aliases", `{"contents":[{"parts":[{"function_call":{"name":"f","args":{"n":9007199254740993,"thought_signature":"business"}},"thought_signature":"old"}]}]}`, `{"contents":[{"parts":[{"function_call":{"name":"f","args":{"n":9007199254740993,"thought_signature":"business"}},"thoughtSignature":"skip_thought_signature_validator"}]}]}`, true},
		{"marked carriers", `{"contents":[{"parts":[{"thoughtSignature":"a"},{"text":"","thought":true}]},{"role":"user","parts":[{"text":"keep"}]}]}`, `{"contents":[{"role":"user","parts":[{"text":"keep"}]}]}`, true},
		{"preserve empty and null", `{"contents":[{"parts":null,"role":"user"},{"parts":[]},{"parts":[{"text":"keep","thoughtSignature":"old"}]}]}`, `{"contents":[{"parts":null,"role":"user"},{"parts":[]},{"parts":[{"text":"keep"}]}]}`, true},
		{"other fields", `{"contents":[{"parts":[{"thoughtSignature":"old","payload":{"n":17},"thought":false}]}],"systemInstruction":{"parts":[{"text":"keep","thoughtSignature":"outside"}]}}`, `{"contents":[{"parts":[{"payload":{"n":17},"thought":false}]}],"systemInstruction":{"parts":[{"text":"keep","thoughtSignature":"outside"}]}}`, true},
		{"unmarked", `{"contents":[{"parts":[{}, {"text":""}, {"thought":false}]}]}`, "", false},
		{"nothing remains", `{"contents":[{"parts":[{"text":"","thought_signature":"a"}]}]}`, "", false},
		{"invalid", `{"contents":`, "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, changed := sanitizeGeminiNativeThoughtSignatures([]byte(tc.input), false, true)
			require.Equal(t, tc.changed, changed)
			if !changed {
				require.Equal(t, tc.input, string(out))
				return
			}
			require.JSONEq(t, tc.want, string(out))
			if strings.Contains(tc.input, "9007199254740993") {
				require.Contains(t, string(out), "9007199254740993")
			}
		})
	}
	for _, force := range []bool{false, true} {
		cleaned, _ := sanitizeGeminiNativeThoughtSignatures([]byte(nativeRectifierRequest), false, force)
		repeated, changed := sanitizeGeminiNativeThoughtSignatures(cleaned, false, force)
		require.False(t, changed)
		require.Equal(t, cleaned, repeated)
	}
}

func TestNativeRectifierEnvelope(t *testing.T) {
	bare := append([]byte{0x01, 0x8f, 0x3d, 0x6b, 0x5f}, make([]byte, 16)...)
	for _, tc := range []struct {
		name, signature              string
		allowBare, allowBypass, want bool
	}{
		{"wrapped", nativeRectifierWrappedSignature, false, false, true},
		{"trimmed unpadded", " \t" + strings.TrimRight(nativeRectifierWrappedSignature, "=") + "\n", false, false, true},
		{"bare on Vertex", base64.StdEncoding.EncodeToString(bare), true, false, true},
		{"bare on relay", base64.StdEncoding.EncodeToString(bare), false, false, false},
		{"short bare", base64.StdEncoding.EncodeToString(bare[:20]), true, false, false},
		{"unknown key", base64.StdEncoding.EncodeToString(append([]byte{0x01, 0, 0, 0, 0}, bare[5:]...)), true, false, false},
		{"outer trailing bytes", base64.StdEncoding.EncodeToString([]byte{0x12, 3, 0x0a, 1, 'x', 0}), false, false, false},
		{"inner trailing bytes", base64.StdEncoding.EncodeToString([]byte{0x12, 4, 0x0a, 1, 'x', 0}), false, false, false},
		{"truncated", base64.StdEncoding.EncodeToString([]byte{0x12, 4, 0x0a, 1, 'x'}), false, false, false},
		{"wrong outer field", base64.StdEncoding.EncodeToString([]byte{0x0a, 3, 0x0a, 1, 'x'}), false, false, false},
		{"empty value", base64.StdEncoding.EncodeToString([]byte{0x12, 2, 0x0a, 0}), false, false, false},
		{"invalid base64", "invalid_signature", false, false, false},
		{"empty", "", false, false, false},
		{"bypass outside first call", geminiDummyThoughtSignature, false, false, false},
		{"trimmed first-call bypass", " context_engineering_is_the_way_to_go ", false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := ""
			if tc.want {
				want = strings.TrimSpace(tc.signature)
			}
			require.Equal(t, want, replayGeminiNativeThoughtSignature(tc.signature, tc.allowBare, tc.allowBypass))
		})
	}
}

func TestNativeRectifierVertexTarget(t *testing.T) {
	for _, tc := range []struct {
		name, accountType, baseURL, projectID, authMode, action string
		want                                                    bool
	}{
		{"default", AccountTypeAPIKey, "", "", "", "generateContent", false},
		{"relay", AccountTypeAPIKey, "https://relay.example.test", "", "", "generateContent", false},
		{"global", AccountTypeAPIKey, "https://AIPLATFORM.googleapis.com:443", "", "", "generateContent", true},
		{"regional", AccountTypeAPIKey, "https://us-central1-aiplatform.googleapis.com", "", "", "generateContent", true},
		{"multi US", AccountTypeAPIKey, "https://aiplatform.us.rep.googleapis.com", "", "", "generateContent", true},
		{"multi EU OAuth", AccountTypeOAuth, "https://aiplatform.eu.rep.googleapis.com", "", "", "generateContent", true},
		{"Code Assist ignores custom URL", AccountTypeOAuth, "https://aiplatform.googleapis.com", "project", "", "generateContent", false},
		{"OAuth count uses custom URL", AccountTypeOAuth, "https://aiplatform.googleapis.com", "project", "", "countTokens", true},
		{"service account", AccountTypeServiceAccount, "", "project", "", "generateContent", true},
		{"Vertex Express count", AccountTypeServiceAccount, "", "", "apikey", "countTokens", true},
		{"Vertex project key", AccountTypeServiceAccount, "", "project", "apikey", "generateContent", true},
		{"path spoof", AccountTypeAPIKey, "https://relay.example.test/aiplatform.googleapis.com", "", "", "generateContent", false},
		{"host spoof", AccountTypeAPIKey, "https://aiplatform.googleapis.com.evil.test", "", "", "generateContent", false},
		{"empty region", AccountTypeAPIKey, "https://-aiplatform.googleapis.com", "", "", "generateContent", false},
		{"invalid URL", AccountTypeAPIKey, "https://%", "", "", "generateContent", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			account := &Account{Platform: PlatformGemini, Type: tc.accountType, Credentials: map[string]any{
				"base_url": tc.baseURL, "project_id": tc.projectID, "auth_mode": tc.authMode,
			}}
			require.Equal(t, tc.want, isGeminiNativeVertexTarget(account, tc.action))
		})
	}
}

func TestNativeRectifierGrouping(t *testing.T) {
	good := nativeRectifierWrappedSignature
	call := func(signature string) string {
		return `{"functionCall":{"name":"f"}` + signature + `}`
	}
	signed := call(`,"thoughtSignature":"` + good + `"`)
	bad := call(`,"thoughtSignature":"bad"`)
	unsigned := call("")
	dummy := call(`,"thoughtSignature":"skip_thought_signature_validator"`)
	for _, tc := range []struct {
		name, parts, want string
		force             bool
	}{
		{"signed siblings", signed + "," + signed, signed + "," + signed, false},
		{"bad siblings", bad + "," + bad, dummy + "," + unsigned, false},
		{"mixed siblings", signed + "," + bad, signed + "," + unsigned, false},
		{"missing siblings", unsigned + "," + unsigned, dummy + "," + unsigned, false},
		{"text separator", bad + `,{"text":"next"},` + bad, dummy + `,{"text":"next"},` + dummy, false},
		{"unsigned thought keeps group", bad + `,{"thought":true,"text":"thinking"},` + unsigned, dummy + `,{"thought":true,"text":"thinking"},` + unsigned, false},
		{"deleted carrier", bad + `,{"thoughtSignature":"bad"},` + bad, dummy + "," + unsigned, false},
		{"response keeps group", bad + `,{"functionResponse":{"name":"f"},"thoughtSignature":"bad"},` + bad, dummy + `,{"functionResponse":{"name":"f"}},` + unsigned, false},
		{"force thought separator", signed + `,{"thought":true,"text":"thinking"},` + signed, dummy + `,{"text":"thinking"},` + dummy, true},
		{"force deleted thought carrier", signed + `,{"thought":true,"text":""},` + signed, dummy + "," + unsigned, true},
		{"bypass siblings", call(`,"thoughtSignature":"context_engineering_is_the_way_to_go"`) + "," + dummy, call(`,"thoughtSignature":"context_engineering_is_the_way_to_go"`) + "," + unsigned, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := []byte(`{"contents":[{"role":"model","parts":[` + tc.parts + `]}]}`)
			out, _ := sanitizeGeminiNativeThoughtSignatures(body, false, tc.force)
			require.JSONEq(t, `{"contents":[{"role":"model","parts":[`+tc.want+`]}]}`, string(out))
		})
	}
}

func TestNativeRectifierPartBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name, input, want string
		force             bool
	}{
		{"snake call and signature", `{"contents":[{"parts":[{"function_call":{"name":"f"},"thought_signature":" EgMKAXg= "}]}]}`, `{"contents":[{"parts":[{"function_call":{"name":"f"},"thoughtSignature":"EgMKAXg="}]}]}`, false},
		{"canonical signature wins", `{"contents":[{"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"bad","thought_signature":"EgMKAXg="}]}]}`, `{"contents":[{"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"skip_thought_signature_validator"}]}]}`, false},
		{"user and response", `{"contents":[{"role":"user","parts":[{"text":"keep","thoughtSignature":"bad"},{"function_response":{"response":{"n":9007199254740993}},"thought_signature":"bad","thought":true}]}]}`, `{"contents":[{"role":"user","parts":[{"text":"keep","thoughtSignature":"bad"},{"function_response":{"response":{"n":9007199254740993}},"thought":true}]}]}`, false},
		{"force response", `{"contents":[{"role":"function","parts":[{"functionResponse":{"name":"f"},"thoughtSignature":"bad","thought":true}]}]}`, `{"contents":[{"role":"function","parts":[{"functionResponse":{"name":"f"}}]}]}`, true},
		{"invalid roles stay local", `{"contents":[{"role":null,"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"bad"},{"functionResponse":{"name":"f"},"thoughtSignature":"bad"}]},{"role":17,"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"bad"}]},{"role":"model","parts":[{"functionCall":{"name":"g"}}]}]}`, `{"contents":[{"role":null,"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"bad"},{"functionResponse":{"name":"f"}}]},{"role":17,"parts":[{"functionCall":{"name":"f"},"thoughtSignature":"bad"}]},{"role":"model","parts":[{"functionCall":{"name":"g"},"thoughtSignature":"skip_thought_signature_validator"}]}]}`, false},
		{"true thought with bad signature", `{"contents":[{"parts":[{"text":"thinking","thought":true,"thoughtSignature":"bad"}]}]}`, `{"contents":[{"parts":[{"text":"thinking","thought":true}]}]}`, false},
		{"function thought flag", `{"contents":[{"parts":[{"functionCall":{"name":"f"},"thought":true,"thoughtSignature":"bad"}]}]}`, `{"contents":[{"parts":[{"functionCall":{"name":"f"},"thought":true,"thoughtSignature":"skip_thought_signature_validator"}]}]}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, changed := sanitizeGeminiNativeThoughtSignatures([]byte(tc.input), false, tc.force)
			require.True(t, changed)
			require.JSONEq(t, tc.want, string(out))
		})
	}
	for _, key := range []string{"toolCall", "tool_call", "toolResponse", "tool_response"} {
		body := []byte(`{"contents":[{"parts":[{"` + key + `":{"name":"f"},"thoughtSignature":"bad","thought":true}]}]}`)
		out, changed := sanitizeGeminiNativeThoughtSignatures(body, false, true)
		require.False(t, changed, key)
		require.Equal(t, body, out)
	}
	for _, input := range []string{
		`{"contents":{}}`,
		`{"contents":[{"parts":[{"text":"keep"}]},{"parts":"invalid"}]}`,
	} {
		out, changed := sanitizeGeminiNativeThoughtSignatures([]byte(input), false, false)
		require.False(t, changed)
		require.Equal(t, input, string(out))
	}
}

func TestNativeRectifierPreflight(t *testing.T) {
	bare := base64.StdEncoding.EncodeToString(append([]byte{0x01, 0x8f, 0x3d, 0x6b, 0x5f}, make([]byte, 16)...))
	request := `{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f"},"thoughtSignature":"` + bare + `"},{"functionCall":{"name":"g"}},{"text":"keep","thoughtSignature":"` + bare + `"}]}]}`
	for _, tc := range []struct {
		name, baseURL string
		disabled      bool
		preserve      bool
	}{
		{"relay", "https://relay.example.test", false, false},
		{"Vertex custom URL", "https://aiplatform.googleapis.com", false, true},
		{"disabled", "https://relay.example.test", true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, upstream := newNativeRectifierService(nativeRectifierResponse{status: 200, body: nativeRectifierSuccess})
			if tc.disabled {
				svc.rateLimitService.settingService = NewSettingService(&nativeRectifierSettingRepo{value: `{"enabled":true,"gemini_native_signature_enabled":false}`}, svc.cfg)
			}
			account := geminiDefaultPolicyAPIKeyAccount()
			account.Credentials["base_url"] = tc.baseURL
			c, _ := nativeRectifierContext()
			_, err := svc.ForwardNative(context.Background(), c, account, "gemini-2.5-flash", "generateContent", false, []byte(request))
			require.NoError(t, err)
			require.Len(t, upstream.requests, 1)
			require.Empty(t, nativeRectifierKinds(c))
			if tc.preserve {
				require.Contains(t, string(upstream.bodies[0]), bare)
			} else {
				require.NotContains(t, string(upstream.bodies[0]), bare)
			}
			wantDummies := 1
			if tc.preserve && !tc.disabled {
				wantDummies = 0
			}
			require.Equal(t, wantDummies, strings.Count(string(upstream.bodies[0]), geminiDummyThoughtSignature))
		})
	}
}

func TestNativeRectifierAllBuilders(t *testing.T) {
	for _, kind := range []string{"relay", "oauth", "codeassist", "vertex"} {
		t.Run(kind, func(t *testing.T) {
			success := nativeRectifierSuccess
			account := &Account{ID: 701, Platform: PlatformGemini, Type: AccountTypeAPIKey, Concurrency: 1, Credentials: map[string]any{"api_key": "synthetic-key", "base_url": "https://relay.example.test"}}
			if kind == "oauth" || kind == "codeassist" {
				account.Type = AccountTypeOAuth
				account.Credentials = map[string]any{"access_token": "synthetic-token"}
			}
			if kind == "codeassist" {
				account.Credentials["project_id"] = "synthetic-project"
				success = "data: {\"response\":" + nativeRectifierSuccess + "}\n\n"
			}
			if kind == "vertex" {
				account = vertexServiceAccount()
				account.ID = 701
			}
			svc, u := newNativeRectifierService(nativeRectifierResponse{status: 400, body: nativeRectifierRejected}, nativeRectifierResponse{status: 200, body: success})
			c, rec := nativeRectifierContext()
			result, err := svc.ForwardNative(context.Background(), c, account, "gemini-2.5-flash", "generateContent", false, []byte(nativeRectifierRequest))
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, 200, rec.Code)
			require.Len(t, u.requests, 2)
			require.Equal(t, u.requests[0].URL.String(), u.requests[1].URL.String())
			require.Equal(t, u.requests[0].Header, u.requests[1].Header)
			require.Contains(t, string(u.bodies[0]), nativeRectifierWrappedSignature)
			require.NotContains(t, string(u.bodies[1]), nativeRectifierWrappedSignature)
			require.Contains(t, string(u.bodies[1]), "skip_thought_signature_validator")
			require.Contains(t, string(u.bodies[1]), `"thoughtSignature":"business"`)
			if kind == "codeassist" {
				require.Contains(t, u.requests[1].URL.Path, "v1internal:streamGenerateContent")
				require.Contains(t, string(u.bodies[1]), `"project":"synthetic-project"`)
				require.Contains(t, string(u.bodies[1]), `"request":`)
			}
			require.Equal(t, []string{"signature_error"}, nativeRectifierKinds(c))
			require.Equal(t, "synthetic-1", result.RequestID)
			for _, body := range u.responseBodies {
				require.True(t, body.closed)
			}
		})
	}
}

func TestNativeRectifierCounting(t *testing.T) {
	for _, status := range []int{200, 400} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			body := `{"totalTokens":1234}`
			if status == 400 {
				body = nativeRectifierRejected
			}
			svc, u := newNativeRectifierService(nativeRectifierResponse{status: 400, body: nativeRectifierRejected}, nativeRectifierResponse{status: status, body: body})
			c, rec := nativeRectifierContext()
			_, err := svc.ForwardNative(context.Background(), c, geminiDefaultPolicyAPIKeyAccount(), "gemini-2.5-flash", "countTokens", false, []byte(nativeRectifierRequest))
			require.Len(t, u.requests, 2)
			require.Equal(t, status, rec.Code)
			if status == 200 {
				require.NoError(t, err)
				require.JSONEq(t, body, rec.Body.String())
			} else {
				require.Error(t, err)
				require.NotContains(t, rec.Body.String(), "totalTokens")
				require.Contains(t, nativeRectifierKinds(c), "signature_retry")
				raw, _ := c.Get(OpsUpstreamErrorsKey)
				events := raw.([]*OpsUpstreamErrorEvent)
				require.Equal(t, "signature_error", events[0].Kind)
				require.Equal(t, 400, events[0].UpstreamStatusCode)
				require.Equal(t, "synthetic-0", events[0].UpstreamRequestID)
				require.Equal(t, "signature_retry", events[1].Kind)
				require.Equal(t, 400, events[1].UpstreamStatusCode)
				require.Equal(t, "synthetic-1", events[1].UpstreamRequestID)
			}
		})
	}
}

func TestNativeRectifierStreaming(t *testing.T) {
	body := "data: " + nativeRectifierSuccess + "\n\n"
	svc, _ := newNativeRectifierService(
		nativeRectifierResponse{status: 400, body: nativeRectifierRejected},
		nativeRectifierResponse{status: 200, body: body},
	)
	c, recorder := nativeRectifierContext()
	result, err := svc.ForwardNative(context.Background(), c, geminiDefaultPolicyAPIKeyAccount(), "gemini-2.5-flash", "streamGenerateContent", true, []byte(nativeRectifierRequest))
	require.NoError(t, err)
	require.True(t, result.Stream)
	require.Contains(t, recorder.Body.String(), `"text":"ok"`)
}

func TestNativeRectifierResendChecksFreshPolicy(t *testing.T) {
	svc, upstream := newNativeRectifierService(
		nativeRectifierResponse{status: 400, body: nativeRectifierRejected},
		nativeRectifierResponse{status: 401, body: geminiSingleCheck401Body},
	)
	repo := &geminiSingleCheckRepoStub{}
	svc.accountRepo = repo
	svc.rateLimitService = NewRateLimitService(repo, nil, svc.cfg, nil, nil)
	svc.rateLimitService.settingService = NewSettingService(&nativeRectifierSettingRepo{value: `{"enabled":true,"gemini_native_signature_enabled":true}`}, svc.cfg)
	c, _ := nativeRectifierContext()
	_, err := svc.ForwardNative(context.Background(), c, geminiAPIKeyAccountWithTempRule(geminiAccountScoped401Rule()), "gemini-2.5-flash", "generateContent", false, []byte(nativeRectifierRequest))
	var failover *UpstreamFailoverError
	require.ErrorAs(t, err, &failover)
	require.Equal(t, 401, failover.StatusCode)
	require.Len(t, upstream.requests, 2)
	require.Equal(t, 1, repo.tempCalls)
	require.Zero(t, repo.setErrCalls)
	require.Equal(t, []string{"signature_error", "signature_retry", "failover"}, nativeRectifierKinds(c))
}

func TestNativeRectifierRecoveryGuards(t *testing.T) {
	for _, tc := range []struct {
		name, body, response string
		status               int
		disabled, policy     bool
	}{
		{"disabled", nativeRectifierRequest, nativeRectifierRejected, 400, true, false},
		{"unchanged", `{"contents":[{"parts":[{"text":"hi"}]}]}`, nativeRectifierRejected, 400, false, false},
		{"unrelated", nativeRectifierRequest, `{"error":{"message":"authentication signature invalid"}}`, 400, false, false},
		{"other status", nativeRectifierRequest, nativeRectifierRejected, 422, false, false},
		{"policy", nativeRectifierRequest, nativeRectifierRejected, 400, false, true},
		{"cached content", nativeRectifierRequest, `{"error":{"message":"Invalid resource state for cache content 8079767076522164224. Invalid thought signature."}}`, 400, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc, u := newNativeRectifierService(nativeRectifierResponse{status: tc.status, body: tc.response})
			if tc.disabled {
				svc.rateLimitService.settingService = NewSettingService(&nativeRectifierSettingRepo{value: `{"enabled":true,"gemini_native_signature_enabled":false}`}, svc.cfg)
			}
			account := geminiDefaultPolicyAPIKeyAccount()
			if tc.policy {
				account.Credentials["custom_error_codes_enabled"] = true
				account.Credentials["custom_error_codes"] = []any{float64(422)}
			}
			c, _ := nativeRectifierContext()
			_, err := svc.ForwardNative(context.Background(), c, account, "gemini-2.5-flash", "generateContent", false, []byte(tc.body))
			require.Error(t, err)
			require.Len(t, u.requests, 1)
			require.NotContains(t, nativeRectifierKinds(c), "signature_error")
		})
	}
}

func TestNativeRectifierTransientBudgetAndEvents(t *testing.T) {
	for _, statuses := range [][]int{{503, 400, 200}, {400, 429, 503, 200}} {
		t.Run(fmt.Sprint(statuses), func(t *testing.T) {
			responses := make([]nativeRectifierResponse, 0, len(statuses))
			for _, status := range statuses {
				body := `{"error":{"message":"temporary"}}`
				if status == 400 {
					body = nativeRectifierRejected
				}
				if status == 200 {
					body = nativeRectifierSuccess
				}
				responses = append(responses, nativeRectifierResponse{status: status, body: body})
			}
			svc, u := newNativeRectifierService(responses...)
			svc.cfg.Gateway.MaxSameAccountRetriesGemini = len(statuses) - 2
			c, _ := nativeRectifierContext()
			_, err := svc.ForwardNative(context.Background(), c, geminiDefaultPolicyAPIKeyAccount(), "gemini-2.5-flash", "generateContent", false, []byte(nativeRectifierRequest))
			require.NoError(t, err)
			require.Len(t, u.requests, len(statuses))
			if len(statuses) == 4 {
				require.Equal(t, []string{"signature_error", "signature_retry", "retry", "retry"}, nativeRectifierKinds(c))
				require.Contains(t, string(u.bodies[3]), "skip_thought_signature_validator")
			}
		})
	}
}

func TestNativeRectifierResendFailureAndCancellation(t *testing.T) {
	for _, mode := range []string{"transport", "construction", "cancellation"} {
		t.Run(mode, func(t *testing.T) {
			svc, u := newNativeRectifierService(nativeRectifierResponse{status: 400, body: nativeRectifierRejected}, nativeRectifierResponse{err: errors.New("synthetic transport failure")})
			account := geminiDefaultPolicyAPIKeyAccount()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			u.after = func(index int) {
				if index == 0 && mode == "construction" {
					account.Credentials["api_key"] = ""
				}
				if index == 0 && mode == "cancellation" {
					cancel()
				}
			}
			c, _ := nativeRectifierContext()
			_, err := svc.ForwardNative(ctx, c, account, "gemini-2.5-flash", "generateContent", false, []byte(nativeRectifierRequest))
			require.Error(t, err)
			if mode == "cancellation" {
				require.ErrorIs(t, err, context.Canceled)
				require.Len(t, u.requests, 1)
				require.NotContains(t, nativeRectifierKinds(c), "signature_error")
			} else {
				require.Contains(t, nativeRectifierKinds(c), "signature_retry_request_error")
			}
			require.True(t, u.responseBodies[0].closed)
		})
	}
}

func TestNativeRectifierCancellationDuringSettingsRead(t *testing.T) {
	svc, upstream := newNativeRectifierService()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	svc.rateLimitService.settingService = NewSettingService(&nativeRectifierCanceledSettingRepo{cancel: cancel}, svc.cfg)
	c, recorder := nativeRectifierContext()
	_, err := svc.ForwardNative(ctx, c, geminiDefaultPolicyAPIKeyAccount(), "gemini-2.5-flash", "generateContent", false, []byte(nativeRectifierRequest))
	require.ErrorIs(t, err, context.Canceled)
	require.Empty(t, upstream.requests)
	require.Empty(t, nativeRectifierKinds(c))
	require.Empty(t, recorder.Body.String())
}
