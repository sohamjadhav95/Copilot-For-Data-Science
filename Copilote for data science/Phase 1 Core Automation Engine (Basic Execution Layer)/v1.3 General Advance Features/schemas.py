JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [
            "filter", "aggregate", "sort", 
            "group", "calculate", "add_column",
            "visualize", "join", "clean"
        ]},
        "columns": {"type": "array", "items": {"type": "string"}},
        "conditions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "operator": {"type": "string"},
                "value": {"type": ["number", "string"]}
            }
        }},
        "aggregation": {"type": "object", "properties": {
            "column": {"type": "string"},
            "operation": {"type": "string"}
        }},
        "visualization": {"type": "object", "properties": {
            "type": {"type": "string", "enum": ["bar", "line", "pie", "scatter", "heatmap", "histogram"]},
            "x": {"type": "string"},
            "y": {"type": "string"},
            "hue": {"type": "string"},
            "rows": {"type": "string"},
            "cols": {"type": "string"},
            "value": {"type": "string"},
            "bins": {"type": "number"},
            "title": {"type": "string"}
        }},
        "clean": {
            "type": "object",
            "properties": {
                "remove_duplicates": {"type": "object", "properties": {"subset": {"type": "array", "items": {"type": "string"}}}},
                "fill_missing": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "method": {"type": "string", "enum": ["mean", "median", "mode"]}
                        }
                    }
                },
                "remove_outliers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "method": {"type": "string", "enum": ["mean", "median", "mode"]},
                            "multiplier": {"type": "number"}
                        }
                    }
                },
                "remove_columns": {"type": "object", "properties": {"columns": {"type": "array", "items": {"type": "string"}}}},
                "rename_columns": {
                    "type": "object",
                    "properties": {
                        "old_names": {"type": "array", "items": {"type": "string"}},
                        "new_names": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "reorder_columns": {"type": "object", "properties": {"new_order": {"type": "array", "items": {"type": "string"}}}},
                "add_column": {
                    "type": "object",
                    "properties": {
                        "new_column": {"type": "string"},
                        "formula": {"type": "string"}
                    }
                }
            }
        }
    }
}