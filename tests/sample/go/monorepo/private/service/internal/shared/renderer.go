package shared

import "fmt"

// Renderer is an interface whose methods are called from other packages.
// This mirrors gin's render.Render interface where r.Render and
// r.WriteContentType are called from context.go.
type Renderer interface {
	// Render writes output to the given destination string.
	Render(dest string) error
	// WriteContentType returns the content type string.
	WriteContentType() string
}

// TextRenderer is a concrete implementation of Renderer.
type TextRenderer struct {
	Content string
}

// Render writes the content to the destination.
func (t *TextRenderer) Render(dest string) error {
	fmt.Println(t.Content, "->", dest)
	return nil
}

// WriteContentType returns the content type.
func (t *TextRenderer) WriteContentType() string {
	return "text/plain"
}
