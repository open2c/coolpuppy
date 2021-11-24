import os.path as op


def validate_csv(value, default_column="balanced.avg"):
    if value is None:
        return
    file_path, _, field_name = value.partition("::")
    if not op.exists(file_path):
        raise ValueError(f"Path not found: {file_path}")
    if not field_name:
        field_name = default_column
    elif field_name.isdigit():
        field_name = int(field_name)
    return file_path, field_name
