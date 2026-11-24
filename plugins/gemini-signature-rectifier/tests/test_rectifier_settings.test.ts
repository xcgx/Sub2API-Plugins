import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, shallowMount } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import SettingsView from '@/views/admin/SettingsView.vue'
import en from '@/i18n/locales/en'
import zh from '@/i18n/locales/zh'

const mocks = vi.hoisted(() => ({
  getRectifierSettings: vi.fn(),
  updateRectifierSettings: vi.fn(),
  showError: vi.fn(),
}))

vi.mock('@/api', () => ({
  adminAPI: {
    settings: {
      getSettings: vi.fn().mockResolvedValue({ backend_mode_enabled: false }),
      getWebSearchEmulationConfig: vi.fn().mockResolvedValue({ enabled: false, providers: [] }),
      getAdminApiKey: vi.fn().mockResolvedValue({ exists: false, masked_key: '' }),
      getOverloadCooldownSettings: vi.fn().mockResolvedValue({}),
      getRateLimit429CooldownSettings: vi.fn().mockResolvedValue({}),
      getPanelRateLimitSettings: vi.fn().mockResolvedValue({}),
      getStreamTimeoutSettings: vi.fn().mockResolvedValue({}),
      getBetaPolicySettings: vi.fn().mockResolvedValue({ rules: [] }),
      getRectifierSettings: mocks.getRectifierSettings,
      updateRectifierSettings: mocks.updateRectifierSettings,
    },
    accounts: {
      getUpstreamBillingProbeSettings: vi.fn().mockResolvedValue({ enabled: false }),
      getOllamaCloudUsageSettings: vi.fn().mockResolvedValue({ enabled: false }),
      getOpenCodeGoUsageSettings: vi.fn().mockResolvedValue({ enabled: false }),
    },
    groups: { getAll: vi.fn().mockResolvedValue([]) },
    proxies: { list: vi.fn().mockResolvedValue({ items: [] }) },
    payment: { getProviders: vi.fn().mockResolvedValue({ data: [] }) },
  },
}))

vi.mock('@/stores', () => ({
  useAppStore: () => ({
    showError: mocks.showError,
    showSuccess: vi.fn(),
    fetchPublicSettings: vi.fn(),
  }),
}))

vi.mock('@/stores/adminSettings', () => ({
  useAdminSettingsStore: () => ({ fetch: vi.fn() }),
}))

vi.mock('@/composables/useClipboard', () => ({
  useClipboard: () => ({ copyToClipboard: vi.fn() }),
}))

describe('native Gemini rectifier settings', () => {
  beforeEach(() => vi.clearAllMocks())

  for (const locale of ['en', 'zh'] as const) {
    it(locale === 'en' ? 'saves both boolean states and preserves siblings' : 'shows Chinese labels and ARIA', async () => {
      let saved = {
        enabled: true,
        thinking_signature_enabled: true,
        gemini_native_signature_enabled: false,
        thinking_budget_enabled: false,
        apikey_signature_enabled: true,
        apikey_signature_patterns: ['synthetic-pattern'],
      }
      mocks.getRectifierSettings.mockImplementation(async () => ({ ...saved }))
      mocks.updateRectifierSettings.mockImplementation(async (payload) => {
        saved = { ...payload }
        return { ...saved }
      })
      const mountView = () => shallowMount(SettingsView, {
        global: {
          plugins: [createI18n({ legacy: false, locale, messages: { en, zh } })],
          stubs: { AppLayout: { template: '<div><slot /></div>' }, Toggle: false, RouterLink: true },
        },
      })
      for (const enabled of locale === 'en' ? [true, false] : [undefined]) {
        const wrapper = mountView()
        try {
          await flushPromises()
          await wrapper.get('#settings-tab-gateway').trigger('click')
          const label = locale === 'en'
            ? 'Gemini Native Signature Rectifier'
            : 'Gemini 原生签名整流'
          const hint = locale === 'en'
            ? 'Normalize incompatible thought signatures before native Gemini requests. Retry a rejected request once when error policies allow it.'
            : '发送原生 Gemini 请求前规范化不兼容的思考签名。错误策略允许时，对被拒绝的请求重试一次。'
          expect(wrapper.get('label[for="gemini-native-signature"]').text()).toBe(label)
          expect(wrapper.get('#gemini-native-signature-hint').text()).toBe(hint)
          expect(wrapper.get('#gemini-native-signature').attributes()).toMatchObject({
            role: 'switch',
            'aria-checked': String(saved.gemini_native_signature_enabled),
            'aria-label': label,
            'aria-describedby': 'gemini-native-signature-hint',
          })
          if (enabled === undefined) {
            continue
          }
          await wrapper.get('#gemini-native-signature').trigger('click')
          const card = wrapper.findAll('.card').find(candidate => candidate.find('#gemini-native-signature').exists())!
          await card.get('button.btn-primary').trigger('click')
          await flushPromises()
          expect(mocks.updateRectifierSettings).toHaveBeenLastCalledWith({
            enabled: true,
            thinking_signature_enabled: true,
            gemini_native_signature_enabled: enabled,
            thinking_budget_enabled: false,
            apikey_signature_enabled: true,
            apikey_signature_patterns: ['synthetic-pattern'],
          })
          expect(wrapper.get('#gemini-native-signature').attributes('aria-checked')).toBe(String(enabled))
          expect(mocks.showError).not.toHaveBeenCalled()
        } finally {
          wrapper.unmount()
        }
      }
    })
  }
})
