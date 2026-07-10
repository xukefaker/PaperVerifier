type ProviderConfig = {
  qa_base_url: string;
  qa_api_key?: string;
  qa_model: string;
};

type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

export async function callChatCompletion(config: ProviderConfig, messages: Message[], maxTokens: number) {
  if (!config.qa_base_url.trim() || !config.qa_model.trim()) {
    throw new Error('Paper QA provider is not configured.');
  }
  const response = await fetch(`${config.qa_base_url.replace(/\/+$/, '')}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(config.qa_api_key ? { Authorization: `Bearer ${config.qa_api_key}` } : {}),
    },
    body: JSON.stringify({
      model: config.qa_model,
      messages,
      temperature: 0,
      max_tokens: maxTokens,
    }),
    signal: AbortSignal.timeout(45_000),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const payload = (await response.json()) as { choices?: { message?: { content?: string } }[] };
  const content = payload.choices?.[0]?.message?.content?.trim();
  if (!content) {
    throw new Error('The provider returned an empty answer.');
  }
  return content;
}
