package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"github.com/sajari/regression"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	f, err := os.Open("data/Advertising.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	df := dataframe.ReadCSV(f)

	for _, col := range df.Names() {
		plot_values := make(plotter.Values, df.Nrow())
		for i, val := range df.Col(col).Float() {
			plot_values[i] = val
		}

		p := plot.New()
		p.Title.Text = fmt.Sprintf("Histogram of %s", col)

		hist, err := plotter.NewHist(plot_values, 16)
		if err != nil {
			log.Fatal(err)
		}

		hist.Normalize(1)

		p.Add(hist)

		if err := p.Save(5 * vg.Inch, 5 * vg.Inch, "images/"  + col + "_histogram.png"); err != nil {
			log.Fatal(err)
		}
	}

	trainNum := (4 * df.Nrow()) / 5
	testNum := df.Nrow() / 5
	if trainNum + testNum < df.Nrow() {
		trainNum++
	}

	trainIndex := make([]int, trainNum)
	testIndex := make([]int, testNum)

	for i := 0; i < trainNum; i++ {
		trainIndex[i] = i
	}

	for i := 0; i < testNum; i++ {
		testIndex[i] = trainNum + i
	}

	trainDF := df.Subset(trainIndex)
	testDF := df.Subset(testIndex)

	df_map := map[int]dataframe.DataFrame {
		0: trainDF,
		1: testDF,
	}

	for i, df_name := range[]string{"data/train.csv", "data/test.csv"} {
		df_file, err := os.Create(df_name)
		if err != nil {
			log.Fatal(err)
		}

		w := bufio.NewWriter(df_file)
		if err := df_map[i].WriteCSV(w); err != nil {
			log.Fatal(err)
		}
	}

	f, err = os.Open("data/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 4

	trainData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var reg regression.Regression
	reg.SetVar(0, "TV")
	reg.SetVar(1, "Radio")
	reg.SetVar(2, "Newspaper")
	reg.SetObserved("Sales")

	for i, data := range trainData {
		if i == 0 {
			continue
		}

		tvVal, err := strconv.ParseFloat(data[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		radioVal, err := strconv.ParseFloat(data[1], 64)
		if err != nil {
			log.Fatal(err)
		}

		newsVal, err := strconv.ParseFloat(data[2], 64)
		if err != nil {
			log.Fatal(err)
		}

		salesVal, err := strconv.ParseFloat(data[3], 64)
		if err != nil {
			log.Fatal(err)
		}

		reg.Train(regression.DataPoint(salesVal, []float64{tvVal, radioVal, newsVal}))
	}

	reg.Run()
	fmt.Printf("Regression Formula: %v\n", reg.Formula)

	f, err = os.Open("data/test.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader = csv.NewReader(f)
	reader.FieldsPerRecord = 4

	testData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var mae float64
	for i, data := range testData {
		if i == 0 {
			continue
		}

		tvVal, err := strconv.ParseFloat(data[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		radioVal, err := strconv.ParseFloat(data[1], 64)
		if err != nil {
			log.Fatal(err)
		}

		newsVal, err := strconv.ParseFloat(data[2], 64)
		if err != nil {
			log.Fatal(err)
		}

		y_test, err := strconv.ParseFloat(data[3], 64)
		if err != nil {
			log.Fatal(err)
		}

		y_pred, err := reg.Predict([]float64{tvVal, radioVal, newsVal})

		mae += math.Abs(y_test - y_pred) / float64(len(testData))
	}

	fmt.Printf("MAE = %0.2f\n", mae)
}
