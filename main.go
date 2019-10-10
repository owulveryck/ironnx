package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	xvm "gorgonia.org/gorgonia/x/vm"
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
	err = backend.PopulateExprgraph()
	if err != nil {
		log.Println(err)
	}
	exprgraph, _ := backend.GetExprGraph()
	backend.SetVM(xvm.NewGoMachine(exprgraph))
	start := time.Now()
	err = backend.Run()
	elapsed := time.Since(start)
	log.Printf("Computation time: %v\n", elapsed)
	if err != nil {
		log.Println(err)
	}
	output, err := model.GetOutputTensors()
	if err != nil {
		log.Println(err)
	}

	fmt.Println(output[0])
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
