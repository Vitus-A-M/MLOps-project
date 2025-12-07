import anyio
import dagger

async def main():
    async with dagger.Connection() as client:
        src = client.host().directory(".")
        python = (
            client.container()
            .from_("python:3.11")
            .with_directory("/src", src)
            .with_workdir("/src")
            .exec(["pip3", "install", "-r", "requirements.txt"])
            .exec(["pytest", "-q"])
        )

        result = await python.stdout()
        print(result)

anyio.run(main)
