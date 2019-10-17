package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	xvm "gorgonia.org/gorgonia/x/vm"
)

const (
	HSize = 128
	WSize = 128
)

func main() {
	modelOnnx := flag.String("model", "model.onnx", "the pre-trained model in onnx format")
	imgF := flag.String("img", "car.jpg", "the path to the image")
	flag.Parse()
	b, err := ioutil.ReadFile(*modelOnnx)
	if err != nil {
		log.Fatal(err)
	}

	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	img, err := os.Open(*imgF)
	if err != nil {
		log.Fatal(err)
	}

	inputT, err := GetTensorFromImage(img)
	if err != nil {
		log.Fatal(err)
	}
	img.Close()
	model.SetInput(0, inputT)

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
	fmt.Printf("%e\n", output[0])
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
