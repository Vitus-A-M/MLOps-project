package main

import (
    "context"
    "fmt"

    "dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    client, err := dagger.Connect(ctx)
    if err != nil {
        panic(err)
    }
    defer client.Close()

    src := client.Host().Directory(".")

    python := client.Container().
        From("python:3.12").
        WithDirectory("/src", src).
        WithWorkdir("/src").
        WithExec([]string{"pip3", "install", "-r", "requirements.txt"}).
        WithExec([]string{"pytest", "-q"})

    // Run and capture output
    out, err := python.Stdout(ctx)
    if err != nil {
        panic(err)
    }

    fmt.Println(out)
}

