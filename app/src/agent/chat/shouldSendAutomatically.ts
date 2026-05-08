import {
  getToolName,
  isToolUIPart,
  lastAssistantMessageIsCompleteWithToolCalls,
  type UIMessage,
} from "ai";

import {
  EDIT_PROMPT_NAVIGATION_CANCEL_ERROR,
  EDIT_PROMPT_TOOL_NAME,
} from "@phoenix/agent/tools/playgroundPrompt";

/**
 * Gate AI SDK's automatic tool-result continuation.
 *
 * `addToolOutput` always asks `sendAutomaticallyWhen` whether it should submit
 * the next model request. Most completed tool calls should continue via AI
 * SDK's `lastAssistantMessageIsCompleteWithToolCalls` helper. Some UI-owned
 * tools, however, can complete because the live UI surface disappeared rather
 * than because the user finished the requested action. Those terminal results
 * should update the transcript, but should not make PXI continue unprompted.
 *
 * Extend this by adding narrow predicates for other terminal tool outputs that
 * are UX/lifecycle cancellations rather than actionable results for the model.
 */
export function shouldSendAutomaticallyAfterToolOutput({
  messages,
}: {
  messages: UIMessage[];
}): boolean {
  if (hasPromptEditNavigationCancel(messages)) {
    return false;
  }
  return lastAssistantMessageIsCompleteWithToolCalls({ messages });
}

/**
 * Detects the `edit_prompt_instance` lifecycle cancellation emitted when the playground
 * route unmounts before the user accepts or rejects a proposed prompt edit.
 * This terminal tool result is useful for the transcript, but it should not
 * trigger an automatic follow-up model request because the user did not provide
 * an approval decision or a new instruction.
 */
function hasPromptEditNavigationCancel(messages: UIMessage[]): boolean {
  const message = messages[messages.length - 1];
  if (!message || message.role !== "assistant") {
    return false;
  }
  return message.parts.some((part) => {
    if (!isToolUIPart(part)) {
      return false;
    }
    return (
      getToolName(part) === EDIT_PROMPT_TOOL_NAME &&
      part.state === "output-error" &&
      part.errorText === EDIT_PROMPT_NAVIGATION_CANCEL_ERROR
    );
  });
}
