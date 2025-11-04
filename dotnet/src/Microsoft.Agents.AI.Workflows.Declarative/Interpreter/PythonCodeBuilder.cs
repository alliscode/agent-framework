// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Agents.AI.Workflows.Declarative.CodeGen.Python;

namespace Microsoft.Agents.AI.Workflows.Declarative.Interpreter;

internal sealed class PythonCodeBuilder : IModelBuilder<string>
{
    private readonly HashSet<string> _actions;
    private readonly List<string> _definitions;
    private readonly List<string> _instances;
    private readonly List<string> _edges;
    private readonly string _rootId;

    public PythonCodeBuilder(string rootId)
    {
        this._actions = [];
        this._definitions = [];
        this._instances = [];
        this._edges = [];
        this._rootId = rootId;
    }

    public string GenerateCode(string? workflowNamespace, string? workflowPrefix)
    {
        PythonProviderTemplate template =
            new(this._rootId, this._definitions, this._instances, this._edges)
            {
                Namespace = workflowNamespace,
                Prefix = workflowPrefix,
            };

        return template.TransformText().Trim();
    }

    public void Connect(IModeledAction source, IModeledAction target, string? condition)
    {
        Debug.WriteLine($"> CONNECT: {source.Id} => {target.Id}{(condition is null ? string.Empty : " (?)")}");

        this.HandleAction(source);
        this.HandleAction(target);

        this._edges.Add(new PythonEdgeTemplate(source.Id, target.Id, condition).TransformText());
    }

    private void HandleAction(IModeledAction action)
    {
        // All templates are based on "PythonCodeTemplate"
        if (action is not PythonCodeTemplate template)
        {
            // Something has gone very wrong.
            throw new DeclarativeModelException($"Unable to generate code for: {action.GetType().Name}.");
        }

        if (this._actions.Add(action.Id))
        {
            switch (action)
            {
                case PythonEmptyTemplate:
                case PythonDefaultTemplate:
                    this._instances.Add(template.TransformText());
                    break;
                case PythonActionTemplate actionTemplate:
                    this._definitions.Add(template.TransformText());
                    this._instances.Add(new PythonInstanceTemplate(action.Id, this._rootId, actionTemplate.UseAgentProvider).TransformText());
                    break;
                case PythonRootTemplate:
                    this._definitions.Add(template.TransformText());
                    break;
            }
        }
    }
}
