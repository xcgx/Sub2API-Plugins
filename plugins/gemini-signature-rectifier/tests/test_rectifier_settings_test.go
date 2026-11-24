// sub2api-package: backend/internal/service
package service

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

type nativeRectifierSettingRepo struct {
	SettingRepository
	value string
	err   error
}

func (r *nativeRectifierSettingRepo) GetValue(context.Context, string) (string, error) {
	return r.value, r.err
}

func TestNativeRectifierDefaultAndGate(t *testing.T) {
	require.False(t, DefaultRectifierSettings().GeminiNativeSignatureEnabled)
	for _, tc := range []struct {
		name  string
		value string
		err   error
		want  bool
	}{
		{"missing field", `{"enabled":true}`, nil, false},
		{"master off", `{"enabled":false,"gemini_native_signature_enabled":true}`, nil, false},
		{"option off", `{"enabled":true,"gemini_native_signature_enabled":false,"thinking_signature_enabled":true,"apikey_signature_enabled":true}`, nil, false},
		{"both on ignores unrelated switches", `{"enabled":true,"gemini_native_signature_enabled":true,"thinking_signature_enabled":false,"apikey_signature_enabled":false}`, nil, true},
		{"read error", "", errors.New("synthetic settings failure"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := NewSettingService(&nativeRectifierSettingRepo{value: tc.value, err: tc.err}, nil)
			require.Equal(t, tc.want, svc.IsGeminiNativeSignatureRectifierEnabled(context.Background()))
		})
	}
}
