package main

// Order represents a customer order.
type Order struct {
	Address string
	Amount  float64
}

// Processor is an interface for order processing.
type Processor interface {
	Process() string
}

// NewOrder creates a new Order.
func NewOrder(address string, amount float64) *Order {
	return &Order{
		Address: address,
		Amount:  amount,
	}
}

// Process validates and formats the order.
func (o *Order) Process() string {
	if !ValidateAddress(o.Address) {
		return "invalid address"
	}
	formatted := FormatCurrency(o.Amount)
	return "processed: " + formatted
}

// Apply applies a callback to the order. 日本語コメントでマルチバイトテスト。
func (o *Order) Apply(fn func(*Order)) {
	fn(o)
}

// UnusedFunc is never called by anyone.
func UnusedFunc() {}
