from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass

import logfire
from devtools import debug
from httpx import AsyncClient

from pydantic_ai import Agent, RunContext
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


DOMAIN_AVAILABILITY_API_ENDPOINT = "https://domain-availability.whoisxmlapi.com/api/v1"

@dataclass
class Deps:
    client: AsyncClient
    domain_api_key: str | None


domain_agent = Agent(
    'ollama:llama3.2',
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        'Be concise, reply with one sentence.'
        'Use the `get_domain_availability` tool to check domain availability'
        'help the user find an alternative domain name if the domain is not available'
    ),
    deps_type=Deps,
    retries=2,
)

@dataclass
class DomainInfo:
    domainAvailability: "UNAVAILABLE" | "AVAILABLE"
    domainName: str


@domain_agent.tool
async def get_domain_availability(ctx: RunContext[Deps], domain_names: list[str]) -> list[dict]:
    if ctx.deps.domain_api_key is None:
        # if no API key is provided, return dummy responses for all domains
        return [{'domainAvailability': 'UNAVAILABLE', 'domainName': name} for name in domain_names]
    
    results = []
    for domain_name in domain_names:
        params = {
            'domainName': domain_name,
            'credits': "DA",
            'apiKey': ctx.deps.domain_api_key
        }
        with logfire.span('calling domain availability API', params=params) as span:
            r = await ctx.deps.client.get(DOMAIN_AVAILABILITY_API_ENDPOINT, params=params)
            r.raise_for_status()
            data = r.json()
            span.set_attribute('response', data)
            results.append(data)

    return results


async def main():
    async with AsyncClient() as client:
        # create a free API key at https://domain-availability.whoisxmlapi.com/api
        domain_api_key = os.getenv('DOMAIN_API_KEY')
        deps = Deps(
            client=client, domain_api_key=domain_api_key
        )
        
        result = await domain_agent.run(
            'Is the domain brockbuilds.net available?', deps=deps
        )
        print('Response:', result.data, result.new_messages())

        result = await domain_agent.run(
            'Ah ok, can you suggest 5 alternative domain names for a new portfolio website and check if they are available.',
            deps=deps,
            message_history=result.new_messages()
        )
        print('Response:', result.data)
  


if __name__ == '__main__':
    asyncio.run(main())