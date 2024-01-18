from typing import Any, Dict, Union

class EasyDict(Dict[str, Any]):
    """
    A dictionary subclass that allows attribute-style access to its keys.

    This class extends the standard dictionary to allow accessing its keys
    as if they were attributes. This provides a more convenient and intuitive
    syntax for accessing and modifying dictionary elements.

    Attributes:
        Inherits all attributes from the standard dictionary.
    """

    def __getattr__(self, name: str) -> Any:
        """
        Override the __getattr__ method to allow attribute-style access.

        This method is called when an attribute lookup has not found the attribute
        in the usual places (i.e., it's not an instance attribute nor is it found
        in the class tree for self).

        Args:
            name (str): The attribute name being accessed.

        Returns:
            Any: The value associated with the key 'name' in the dictionary.

        Raises:
            AttributeError: If the key 'name' does not exist in the dictionary.
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{name}' attribute does not exist.")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override the __setattr__ method to allow attribute-style assignment.

        This method is invoked when an attribute assignment is attempted.

        Args:
            name (str): The attribute name where the value will be set.
            value (Any): The value to be set for the given attribute name.
        """
        self[name] = value

    def __delattr__(self, name: str) -> None:
        """
        Override the __delattr__ method to allow attribute-style deletion.

        This method is invoked when an attribute deletion is attempted.

        Args:
            name (str): The attribute name to be deleted.

        Raises:
            AttributeError: If the key 'name' does not exist in the dictionary.
        """
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{name}' attribute does not exist and cannot be deleted.")
