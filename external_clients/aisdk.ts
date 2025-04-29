import {
  CoreAssistantMessage,
  CoreMessage,
  CoreSystemMessage,
  CoreTool,
  CoreUserMessage,
  generateObject,
  generateText,
  ImagePart,
  LanguageModel,
  TextPart,
} from "ai";
import {
  CreateChatCompletionOptions,
  LLMClient,
  AvailableModel,
} from "@browserbasehq/stagehand";
import { ChatCompletion } from "openai/resources";

export class AISdkClient extends LLMClient {
  public type = "aisdk" as const;
  private model: LanguageModel;
  public hasVision = false;
  public clientOptions: any = {};
  public modelName: AvailableModel;

  constructor({ model }: { model: LanguageModel }) {
    super(model.modelId as AvailableModel);
    this.model = model;
    this.modelName = model.modelId as AvailableModel;
  }

  async createChatCompletion<T = ChatCompletion>({
    options,
  }: CreateChatCompletionOptions): Promise<T> {
    const formattedMessages: CoreMessage[] = (options.messages as any[]).map(
      (message): CoreMessage => {
        if (Array.isArray(message.content)) {
          if (message.role === "system") {
            const systemMessage: CoreSystemMessage = {
              role: "system",
              content: message.content
                .map((c: ImagePart | TextPart) =>
                  "text" in c ? c.text ?? "" : ""
                )
                .join("\n"),
            };
            return systemMessage;
          }

          const contentParts = message.content.map(
            (content: ImagePart | TextPart) => {
              if (content.type === "image") {
                const imageContent: ImagePart = {
                  type: "image",
                  image: content.image ?? "",
                };
                return imageContent;
              } else {
                const textContent: TextPart = {
                  type: "text",
                  text: content.text ?? "",
                };
                return textContent;
              }
            }
          );

          if (message.role === "user") {
            const userMessage: CoreUserMessage = {
              role: "user",
              content: contentParts,
            };
            return userMessage;
          } else {
            const textOnlyParts = contentParts.filter(
              (part: ImagePart | TextPart): part is TextPart =>
                part.type === "text"
            );
            const assistantMessage: CoreAssistantMessage = {
              role: "assistant",
              content: textOnlyParts,
            };
            return assistantMessage;
          }
        } else {
          const simpleMessage: CoreMessage = {
            role: message.role,
            content: message.content ?? "",
          };
          return simpleMessage;
        }
      }
    );

    if (options.response_model) {
      const response = await generateObject({
        model: this.model,
        messages: formattedMessages,
        schema: options.response_model.schema,
      });

      return {
        data: response.object,
        usage: {
          prompt_tokens: response.usage.promptTokens ?? 0,
          completion_tokens: response.usage.completionTokens ?? 0,
          total_tokens: response.usage.totalTokens ?? 0,
        },
      } as T;
    }

    const tools: Record<string, CoreTool> = {};

    for (const rawTool of options.tools ?? []) {
      tools[rawTool.name] = {
        description: rawTool.description,
        parameters: rawTool.parameters,
      };
    }

    const response = await generateText({
      model: this.model,
      messages: formattedMessages,
      tools,
    });

    return {
      data: response.text,
      usage: {
        prompt_tokens: response.usage.promptTokens ?? 0,
        completion_tokens: response.usage.completionTokens ?? 0,
        total_tokens: response.usage.totalTokens ?? 0,
      },
    } as T;
  }
}
