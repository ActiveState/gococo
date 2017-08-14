/*
gococo - CLI for executing COCO TensorFlow graphs

Copyright (c) 2017 ActiveState Software

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/colornames"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// Global labels array
var labels []string

// DRAWING UTILITY FUNCTIONS

// HLine draws a horizontal line
func HLine(img *image.RGBA, x1, y, x2 int, col color.Color) {
	for ; x1 <= x2; x1++ {
		img.Set(x1, y, col)
	}
}

// VLine draws a veritcal line
func VLine(img *image.RGBA, x, y1, y2 int, col color.Color) {
	for ; y1 <= y2; y1++ {
		img.Set(x, y1, col)
	}
}

// Rect draws a rectangle utilizing HLine() and VLine()
func Rect(img *image.RGBA, x1, y1, x2, y2, width int, col color.Color) {
	for i := 0; i < width; i++ {
		HLine(img, x1, y1+i, x2, col)
		HLine(img, x1, y2+i, x2, col)
		VLine(img, x1+i, y1, y2, col)
		VLine(img, x2+i, y1, y2, col)
	}
}

// TENSOR UTILITY FUNCTIONS
func makeTensorFromImage(filename string) (*tf.Tensor, image.Image, error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}

	r := bytes.NewReader(b)
	img, _, err := image.Decode(r)

	if err != nil {
		return nil, nil, err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		return nil, nil, err
	}
	// Creates a tensorflow graph to decode the jpeg image
	graph, input, output, err := decodeJpegGraph()
	if err != nil {
		return nil, nil, err
	}
	// Execute that graph to decode this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, err
	}
	return normalized[0], img, nil
}

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func loadLabels(labelsFile string) {
	file, err := os.Open(labelsFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Printf("ERROR: failed to read %s: %v", labelsFile, err)
	}
}

func getLabel(idx int, probabilities []float32, classes []float32) string {
	index := int(classes[idx])
	label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

	return label
}

func addLabel(img *image.RGBA, x, y, class int, label string) {
	col := colornames.Map[colornames.Names[class]]
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(colornames.Black),
		Face: basicfont.Face7x13,
		Dot:  point,
	}

	Rect(img, x, y-13, (x + len(label)*7), y-6, 7, col)

	d.DrawString(label)
}

func main() {
	// Parse flags
	modeldir := flag.String("dir", "", "Directory containing COCO trained model files. Assumes model file is called frozen_inference_graph.pb")
	jpgfile := flag.String("jpg", "", "Path of a JPG image to use for input")
	outjpg := flag.String("out", "output.jpg", "Path of output JPG for displaying labels. Default is output.jpg")
	labelfile := flag.String("labels", "labels.txt", "Path to file of COCO labels, one per line")
	flag.Parse()
	if *modeldir == "" || *jpgfile == "" {
		flag.Usage()
		return
	}

	// Load the labels
	loadLabels(*labelfile)

	// Load a frozen graph to use for queries

	modelpath := filepath.Join(*modeldir, "frozen_inference_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		log.Fatal(err)
	}

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, i, err := makeTensorFromImage(*jpgfile)
	if err != nil {
		log.Fatal(err)
	}

	// Transform the decoded YCbCr JPG image into RGBA
	b := i.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, i, b.Min, draw.Src)

	// Get all the input and output operations
	inputop := graph.Operation("image_tensor")
	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")

	// Execute COCO Graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		log.Fatal(err)
	}

	// Outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Draw a box around the objects
	curObj := 0

	// 0.4 is an arbitrary threshold, below this the results get a bit random
	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		Rect(img, int(x1), int(y1), int(x2), int(y2), 4, colornames.Map[colornames.Names[int(classes[curObj])]])
		addLabel(img, int(x1), int(y1), int(classes[curObj]), getLabel(curObj, probabilities, classes))

		curObj++
	}

	// Output JPG file
	outfile, err := os.Create(*outjpg)
	if err != nil {
		log.Fatal(err)
	}

	var opt jpeg.Options

	opt.Quality = 80

	err = jpeg.Encode(outfile, img, &opt)
	if err != nil {
		log.Fatal(err)
	}
}
