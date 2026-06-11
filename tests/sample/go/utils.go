package main

import "fmt"

// FormatCurrency formats an amount as currency.
func FormatCurrency(amount float64) string {
	return fmt.Sprintf("$%.2f", amount)
}

func NoDocFunc() {}
