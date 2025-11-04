// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Bot.ObjectModel;

namespace Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

internal partial class PythonParseValueTemplate : PythonActionTemplate
{
    public PythonParseValueTemplate(ParseValue model)
    {
        this.Model = this.Initialize(model);
    }

    public ParseValue Model { get; }
}
