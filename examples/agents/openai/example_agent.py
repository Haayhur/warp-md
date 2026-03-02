"""
warp-md OpenAI Agents SDK Example

This example demonstrates how to use warp-md with the OpenAI Agents SDK.

Prerequisites:
    pip install openai openai-agents warp-md

Usage:
    python example_agent.py --topology protein.pdb --trajectory traj.xtc

Reference:
    https://github.com/openai/openai-agents
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent))
from warp_md_tool import WarpMDTool

load_dotenv()


async def main():
    """Run OpenAI agent with warp-md tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-md OpenAI Agents SDK Example"
    )
    parser.add_argument(
        "--topology", "-t",
        required=True,
        help="Path to topology file",
    )
    parser.add_argument(
        "--trajectory", "-tr",
        required=True,
        help="Path to trajectory file",
    )
    parser.add_argument(
        "--query", "-q",
        default="Calculate the radius of gyration for the protein",
        help="Analysis query",
    )

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    # Create client and tool
    client = AsyncOpenAI()
    tool = WarpMDTool()

    # Build messages
    messages = [
        {
            "role": "system",
            "content": """You are a molecular dynamics analyst. Use the warp_md_analysis tool
            to perform trajectory analyses. You have access to these files:
            - Topology: {args.topology}
            - Trajectory: {args.trajectory}

            When you perform analyses, report the results including output file paths.""".replace(
                "{args.topology}", args.topology
            ).replace(
                "{args.trajectory}", args.trajectory
            ),
        },
        {
            "role": "user",
            "content": args.query,
        },
    ]

    # Call OpenAI with tool
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[tool.to_function_definition()],
    )

    # Handle tool calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            if tool_call.function.name == tool.name:
                import json
                func_args = json.loads(tool_call.function.arguments)
                result = tool.call(**func_args)

                print(f"Analysis Result:")
                print(json.dumps(result, indent=2))

                # Follow-up for interpretation
                messages.append(response.choices[0].message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

                interpretation = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )

                print(f"\nInterpretation:")
                print(interpretation.choices[0].message.content)
    else:
        print(response.choices[0].message.content)


if __name__ == "__main__":
    asyncio.run(main())
