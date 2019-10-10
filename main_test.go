package main

import (
	"io/ioutil"
	"log"
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	xvm "gorgonia.org/gorgonia/x/vm"
	"gorgonia.org/tensor"
)

func TestRace(t *testing.T) {
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

	xT := tensor.New(
		tensor.WithShape(1, 128, 128, 3),
		tensor.Of(tensor.Float32))
	model.SetInput(0, xT)
	err = backend.PopulateExprgraph()
	if err != nil {
		log.Println(err)
	}
	exprgraph, _ := backend.GetExprGraph()
	backend.SetVM(xvm.NewGoMachine(exprgraph))
	err = backend.Run()
	if err != nil {
		log.Println(err)
	}
	output, err := model.GetOutputTensors()
	if err != nil {
		log.Println(err)
	}

	log.Println(output[0])
	/*
		b, err = dot.Marshal(exprgraph)
		if err != nil {
			log.Fatal(err)
		}
		err = ioutil.WriteFile("model.dot", b, 0644)
		if err != nil {
			log.Fatal(err)
		}
	*/
}
