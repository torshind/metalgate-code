package main

// UseProcessor calls Process on a Processor interface value.
func UseProcessor(p Processor) string {
	return p.Process()
}
