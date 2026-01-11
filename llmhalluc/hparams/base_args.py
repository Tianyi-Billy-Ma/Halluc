from dataclasses import asdict, dataclass, fields


@dataclass
class BaseArguments:
    @property
    def yaml_exclude(self):
        return set()

    def to_dict(self):
        return asdict(self)

    def to_yaml(
        self,
        exclude: bool = True,
        exclude_keys: set[str] | None = None,
        include_keys: set[str] | None = None,
    ):
        """Convert dataclass to dictionary suitable for YAML serialization.

        Args:
            exclude: If True, exclude fields in yaml_exclude property
            exclude_keys: Additional fields to exclude
            include_keys: Fields to include even if in yaml_exclude

        Returns:
            Dictionary with all fields, properly serialized for YAML
        """
        from pathlib import Path

        exclude_keys = exclude_keys or set()
        include_keys = include_keys or set()

        final_exclude = (
            self.yaml_exclude | exclude_keys - include_keys if exclude else set()
        )

        result = {}
        for f in fields(self):
            # Skip excluded fields
            if f.name in final_exclude:
                continue

            value = getattr(self, f.name)

            # Serialize Path objects to strings for YAML compatibility
            if isinstance(value, Path):
                value = str(value)

            # Include all values, even False, 0, [], None
            # (previous implementation incorrectly filtered these out)
            result[f.name] = value

        return result
