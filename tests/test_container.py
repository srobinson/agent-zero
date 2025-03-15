import pytest
from unittest.mock import Mock, patch, MagicMock, call
import docker
from docker.errors import DockerException, ImageNotFound, APIError

from agents_manager.Container import Container
from agents_manager.Agent import Agent


@pytest.fixture
def mock_docker_client():
    """Fixture that provides a mock Docker client."""
    mock_client = Mock()
    mock_client.images = Mock()
    mock_client.containers = Mock()
    return mock_client


@pytest.fixture
def container(mock_docker_client):
    """Fixture that provides a basic Container instance with a mocked Docker client."""
    with patch("docker.from_env", return_value=mock_docker_client):
        container = Container(
            name="test_container",
            description="Test container",
            image="test/image:latest",
        )
        return container


class TestContainer:
    """Test suite for the Container class."""

    def test_init_with_required_params(self, mock_docker_client):
        """Test initialization with required parameters."""
        with patch("docker.from_env", return_value=mock_docker_client):
            container = Container(
                name="test_container",
                description="Test container",
                image="test/image:latest",
            )

            assert container.name == "test_container"
            assert container.description == "Test container"
            assert container.image == "test/image:latest"
            assert container.environment == {}
            assert container.auth_credentials == {}
            assert container.return_to is None
            assert container.client == mock_docker_client

    # ... other test methods ...

    def test_run_with_return_to_agent(self, container, mock_docker_client):
        """Test container run with return_to agent configuration."""
        # Create a mock agent
        mock_agent = Mock(spec=Agent)

        # Set up return_to configuration in kwargs
        container.kwargs["return_to"] = {
            "agent": mock_agent,
            "instruction": "New instruction with {result}",
        }

        # Mock container run result
        mock_docker_client.containers.run.return_value = b"Container output"

        # Mock the replace_placeholder function
        with patch(
            "agents_manager.Container.replace_placeholder",
            return_value="New instruction with Container output",
        ):
            # Run the container
            result = container.run({"ARG1": "value1"})

            # Check that the agent's instruction was set correctly
            # Use call() to match the exact call pattern
            mock_agent.set_instruction.assert_called_once_with(
                "New instruction with Container output"
            )

            # Check that the result is the agent
            assert result == mock_agent


# This test is outside the TestContainer class
@patch("docker.from_env")
def test_container_integration_with_replace_placeholder(mock_from_env):
    """Test integration with replace_placeholder utility."""
    # Mock the replace_placeholder function
    with patch(
        "agents_manager.Container.replace_placeholder",
        return_value="Process this data: Container output",
    ):

        mock_client = Mock()
        mock_from_env.return_value = mock_client

        # Use bytes for the result
        mock_client.containers.run.return_value = b"Container output"

        # Create a mock agent
        mock_agent = Mock(spec=Agent)

        # Create container with return_to configuration
        container = Container(
            name="test_container",
            description="Test container",
            image="test/image:latest",
            return_to={
                "agent": mock_agent,
                "instruction": "Process this data: {result}",
            },
        )

        # Run the container
        result = container.run({"ARG1": "value1"})

        # Check that the agent's instruction was set correctly
        # Use call() to match the exact call pattern without keyword arguments
        mock_agent.set_instruction.assert_called_once_with(
            "Process this data: Container output"
        )

        # Check that the result is the agent
        assert result == mock_agent
