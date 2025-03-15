from typing import Any, Dict, Optional, Union

import docker
from docker.errors import APIError, DockerException, ImageNotFound

from agentflow.utils import replace_placeholder


class Container:
    """
    A class for running Docker containers as tools for agents.

    This class provides functionality to initialize, authenticate, and run
    Docker containers. It can be used as a tool by agents to execute code
    in isolated environments and return results or even new agents.
    """

    def __init__(self, name: str, description: str, image: str, **kwargs: Any) -> None:
        """
        Initialize a Container with a name, description, image, and optional parameters.

        Args:
            name (str): The name of the container tool.
            description (str): A description of what the container does.
            image (str): The Docker image to use.
            **kwargs (Any): Additional keyword arguments for container configuration.
                - environment (Dict[str, str]): Environment variables to set in the container.
                - authenticate (Dict[str, str]): Authentication credentials for Docker registry.
                - return_to (Dict[str, Any]): Configuration for returning results to an agent.
                - volumes (Dict[str, Dict[str, str]]): Volume mappings.
                - network (str): Docker network to use.
                - command (str): Command to run in the container.

        Raises:
            DockerException: If the Docker client cannot be initialized.
        """
        if not name:
            raise ValueError("Container name cannot be empty")
        if not description:
            raise ValueError("Container description cannot be empty")
        if not image:
            raise ValueError("Container image cannot be empty")

        self.name = name
        self.description = description
        self.image = image
        self.environment = kwargs.get("environment", {})
        self.auth_credentials = kwargs.get("authenticate", {})
        self.return_to = kwargs.get("return_to", None)
        self.kwargs = kwargs
        self.client = None

        # Initialize Docker client
        self.initialize()

        # Authenticate if credentials are provided
        if self.auth_credentials:
            self._authenticate()

    def initialize(self) -> None:
        """
        Initialize the Docker client.

        Raises:
            DockerException: If the Docker client cannot be initialized.
        """
        try:
            self.client = docker.from_env()
        except DockerException as e:
            raise DockerException(f"Failed to initialize Docker client: {e}")

    def _authenticate(self) -> None:
        """
        Authenticate with the Docker registry using provided credentials.

        Raises:
            APIError: If authentication fails.
        """
        if not self.client:
            raise ValueError("Docker client not initialized")

        try:
            self.client.login(
                username=self.auth_credentials.get("username"),
                password=self.auth_credentials.get("password"),
                registry=self.auth_credentials.get("registry"),
            )
        except APIError as e:
            raise APIError(f"Docker registry authentication failed: {e}")

    def pull_image(self) -> None:
        """
        Pull the specified image from the registry.

        Raises:
            ImageNotFound: If the image cannot be found.
            APIError: If there's an error pulling the image.
        """
        if not self.client:
            raise ValueError("Docker client not initialized")

        try:
            self.client.images.pull(self.image)
        except ImageNotFound:
            raise ImageNotFound(f"Image not found: {self.image}")
        except APIError as e:
            raise APIError(f"Error pulling image {self.image}: {e}")

    def run(self, arguments: Dict[str, Any]) -> Any:
        """
        Run the container with provided arguments.

        Args:
            arguments (Dict[str, Any]): Arguments to pass to the container as environment variables.

        Returns:
            Any: The container output or an agent if return_to is configured.

        Raises:
            ValueError: If the Docker client is not initialized.
            DockerException: If there's an error running the container.
        """
        if not self.client:
            raise ValueError("Docker client not initialized")

        # Prepare container configuration
        run_kwargs = self.kwargs.copy()
        run_kwargs["image"] = self.image
        run_kwargs["detach"] = False
        run_kwargs["remove"] = True

        # Merge provided arguments with existing environment variables
        environment = {**self.environment, **arguments}
        run_kwargs["environment"] = environment

        # Extract return_to configuration if present
        return_to = run_kwargs.pop("return_to", None)

        try:
            # Run the container
            result = self.client.containers.run(**run_kwargs)

            # Convert bytes to string if needed
            if isinstance(result, bytes):
                result = result.decode("utf-8")

            # Process return_to if configured
            if return_to and "agent" in return_to:
                if "instruction" in return_to:
                    instruction = replace_placeholder(return_to["instruction"], result)
                    return_to["agent"].set_instruction(instruction)
                return return_to["agent"]

            return result

        except DockerException as e:
            raise DockerException(f"Error running container: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the container to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation of the container.
        """
        return {
            "name": self.name,
            "description": self.description,
            "image": self.image,
            "has_auth": bool(self.auth_credentials),
            "has_return_to": self.return_to is not None,
        }

    def __str__(self) -> str:
        """
        Get a string representation of the container.

        Returns:
            str: String representation of the container.
        """
        return f"Container(name='{self.name}', image='{self.image}')"

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the container.

        Returns:
            str: Detailed string representation of the container.
        """
        return (
            f"Container(name='{self.name}', description='{self.description[:20]}...', "
            f"image='{self.image}')"
        )
