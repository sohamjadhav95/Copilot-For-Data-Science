# schemas.py
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [
            "filter", "aggregate", "sort", 
            "group", "calculate", "add_column",
            "visualize", "join"
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
            "type": {"type": "string", "enum": ["bar", "line", "pie"]},
            "x": {"type": "string"},
            "y": {"type": "string"},
            "title": {"type": "string"}
        }}
    }
}