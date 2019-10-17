package main

import (
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"reflect"

	"github.com/disintegration/gift"
	"gorgonia.org/tensor"
)

// as resize, crop or saturation
func resizeImage(img image.Image) (*image.RGBA, error) {
	resizeFilter := gift.Resize(WSize, HSize, gift.LanczosResampling)
	dst := image.NewRGBA(image.Rect(0, 0, WSize, HSize))
	gift.New(resizeFilter).Draw(dst, img)
	return dst, nil
}

// Create a tensor BHWC from the image; the values are normalized between 0 and 1
func imageToNormalizedBHWC(img *image.RGBA, dst tensor.Tensor) error {
	// check if tensor is a pointer
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("cannot decode image into a non pointer or a nil receiver")
	}
	// check if tensor is compatible with BWHC (4 dimensions)
	if len(dst.Shape()) != 4 {
		return fmt.Errorf("Expected a 4 dimension tensor, but receiver has only %v", len(dst.Shape()))
	}
	// Check the batch size
	if dst.Shape()[0] != 1 {
		return errors.New("only batch size of one is supported")
	}
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	if dst.Shape()[1] != h || dst.Shape()[2] != w {
		return fmt.Errorf("cannot fit image into tensor; image is %v*%v but tensor is %v*%v", h, w, dst.Shape()[2], dst.Shape()[3])
	}
	var m uint8
	for i, e := range img.Pix {
		if i == 0 || e > m {
			m = e
		}
	}
	max := uint32(m)
	max |= max << 8
	switch dst.Dtype() {
	case tensor.Float32:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				r, g, b, a := img.At(x, y).RGBA()
				if a != max && a != 0 {
					return fmt.Errorf("Transparency not handled %v", a)
				}
				err := dst.SetAt(float32(r)/float32(max), 0, y, x, 0)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(g)/float32(max), 0, y, x, 1)
				if err != nil {
					return err
				}
				err = dst.SetAt(float32(b)/float32(max), 0, y, x, 2)
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("%v not handled yet", dst.Dtype())
	}
	return nil
}
func readIMG(r io.Reader) (image.Image, error) {
	return jpeg.Decode(r)
}

// GetTensorFromImage reads an image from r and returns a tensor suitable to run in tiny yolo.
// The tensor is BWHC and is normalized;
// its shape is (1,wSize,hSize,3)
func GetTensorFromImage(r io.Reader) (tensor.Tensor, error) {
	img, err := readIMG(r)
	if err != nil {
		return nil, err
	}
	resized, err := resizeImage(img)
	if err != nil {
		return nil, err
	}
	t := tensor.NewDense(tensor.Float32, []int{1, WSize, HSize, 3})
	err = imageToNormalizedBHWC(resized, t)
	return t, err
}
