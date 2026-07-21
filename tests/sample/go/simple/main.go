package main

import "fmt"

func main() {
	o := NewOrder("123 Main St", 99.95)
	fmt.Println(o.Process())
}
