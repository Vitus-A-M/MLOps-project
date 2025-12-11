package main

import (
    "context"
    "dagger.io/dagger"
)

func main() {
    ctx := context.Background()

    if err := Build(ctx); err != nil {
        panic(err)
    }
}
func Build(ctx context.Context) error {
    client, err := dagger.Connect(ctx)
    if err != nil {
	return err
}
    defer client.Close()


    python := client.Container().From("python:3.14").
        WithDirectory("/mlops_project", client.Host().Directory(".")).
	WithWorkdir("/mlops_project").
        WithExec([]string{"python", "--version"}).
        WithExec([]string{"pip", "install", "-r", "requirements.txt"})

    train := python.
	WithWorkdir("/mlops_project/models").
    	WithExec([]string{"python", "train.py"}).
	WithExec([]string{"mkdir", "-p", "output"})

    python = train.WithExec([]string{"pytest", "-q"})

    _, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

    return nil
}

