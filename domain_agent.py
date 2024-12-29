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
async def get_domain_availability(ctx: RunContext[Deps], domain_name: str) -> { DomainInfo: DomainInfo }:
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

    return data


async def main():
    async with AsyncClient() as client:
        # create a free API key at https://domain-availability.whoisxmlapi.com/api
        domain_api_key = os.getenv('DOMAIN_API_KEY')
        deps = Deps(
            client=client, domain_api_key=domain_api_key
        )
        
        result = await domain_agent.run(
            'Is the domain google.com available?', deps=deps
        )
        print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())