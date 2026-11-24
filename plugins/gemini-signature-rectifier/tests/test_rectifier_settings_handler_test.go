// sub2api-package: backend/internal/handler/admin
package admin

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type nativeRectifierAdminRepo struct {
	service.SettingRepository
	value string
}

func (r *nativeRectifierAdminRepo) GetValue(context.Context, string) (string, error) {
	return r.value, nil
}

func (r *nativeRectifierAdminRepo) Set(_ context.Context, _ string, value string) error {
	r.value = value
	return nil
}

func TestNativeRectifierAdminRoundTrip(t *testing.T) {
	gin.SetMode(gin.TestMode)
	repo := &nativeRectifierAdminRepo{}
	h := NewSettingHandler(service.NewSettingService(repo, nil), nil, nil, nil, nil, nil, nil)
	for _, enabled := range []bool{true, false} {
		input := map[string]any{
			"enabled": true, "gemini_native_signature_enabled": enabled,
			"thinking_signature_enabled": true, "thinking_budget_enabled": false,
			"apikey_signature_enabled": true, "apikey_signature_patterns": []string{"synthetic-pattern"},
		}
		body, err := json.Marshal(input)
		require.NoError(t, err)
		put := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(put)
		c.Request = httptest.NewRequest(http.MethodPut, "/api/v1/admin/settings/rectifier", bytes.NewReader(body))
		c.Request.Header.Set("Content-Type", "application/json")
		h.UpdateRectifierSettings(c)
		require.Equal(t, http.StatusOK, put.Code, put.Body.String())
		get := httptest.NewRecorder()
		c, _ = gin.CreateTestContext(get)
		c.Request = httptest.NewRequest(http.MethodGet, "/api/v1/admin/settings/rectifier", nil)
		h.GetRectifierSettings(c)
		require.Equal(t, http.StatusOK, get.Code)
		for _, response := range []*httptest.ResponseRecorder{put, get} {
			var envelope struct {
				Data map[string]any `json:"data"`
			}
			require.NoError(t, json.Unmarshal(response.Body.Bytes(), &envelope))
			require.Equal(t, enabled, envelope.Data["gemini_native_signature_enabled"])
			require.Equal(t, true, envelope.Data["enabled"])
			require.Equal(t, true, envelope.Data["thinking_signature_enabled"])
			require.Equal(t, false, envelope.Data["thinking_budget_enabled"])
			require.Equal(t, true, envelope.Data["apikey_signature_enabled"])
			require.Equal(t, []any{"synthetic-pattern"}, envelope.Data["apikey_signature_patterns"])
		}
	}
}
