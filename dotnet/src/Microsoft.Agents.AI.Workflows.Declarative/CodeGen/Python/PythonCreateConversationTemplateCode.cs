// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonCreateConversationTemplate : PythonActionTemplate
{
    public PythonCreateConversationTemplate(CreateConversation model)
    {
        this.Model = this.Initialize(model);
    }

    public CreateConversation Model { get; }
}
