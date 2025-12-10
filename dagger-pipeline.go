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
}
func Build(ctx context.Context) error {
    client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
    defer client.Close()


    python := client.Container().From("python:3.12").
        WithDirectory("/mlops_project", client.Host().Directory(".")).
        WithExec([]string{"/python", "--version"})
        WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
    python = WithExec([]string{"pytest", "-q"})
    
    _, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

    return nil
}

