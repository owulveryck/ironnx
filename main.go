package main

import (
	"io/ioutil"
	"log"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

func main() {
	b, err := ioutil.ReadFile("model.onnx")
	if err != nil {
		log.Fatal(err)
	}
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	t := tensor.New(
		tensor.WithShape(1, 128, 128, 3),
		tensor.Of(tensor.Float32))
	model.SetInput(0, t)
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}

}
