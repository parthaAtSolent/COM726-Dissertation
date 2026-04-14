"""Simple connection test to verify graph works"""

from app.core.graph import chatbot
from langchain_core.messages import HumanMessage
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_single_query():
    print("Testing single query...")

    # Simple test
    inputs = {
        "messages": [HumanMessage(content="What is the capital of France?")]
    }

    config = {
        "configurable": {
            "thread_id": "test_connection"
        }
    }

    try:
        result = chatbot.invoke(inputs, config=config)
        print(f"Result type: {type(result)}")
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

        # Try to extract answer
        if isinstance(result, dict):
            messages = result.get("messages", [])
            print(f"Messages count: {len(messages)}")

            if messages:
                last_msg = messages[-1]
                print(f"Last message type: {type(last_msg)}")
                if hasattr(last_msg, 'content'):
                    print(f"Answer: {last_msg.content[:200]}")
                elif isinstance(last_msg, dict):
                    print(
                        f"Answer: {last_msg.get('content', 'No content')[:200]}")

            # Check routing info
            routing = result.get("routing_info", {})
            print(f"Routing info: {routing}")
            route = result.get("route", "No route found")
            print(f"Route: {route}")
        else:
            print(f"Unexpected result format: {result}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_with_state():
    """Test using ChatState directly"""
    print("\nTesting with ChatState...")

    from app.models.chat_state import ChatState

    initial_state: ChatState = {
        "messages": [HumanMessage(content="What is 2+2?")],
        "model": "llama-8b-instant",
        "routing_info": None,
        "rag_context": None,
    }

    config = {
        "configurable": {
            "thread_id": "test_state"
        }
    }

    try:
        result = chatbot.invoke(initial_state, config=config)
        print(f"Result type: {type(result)}")

        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    print(f"Answer: {last_msg.content[:200]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_query()
    test_with_state()
