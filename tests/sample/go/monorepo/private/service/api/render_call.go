package api

import (
	"go.example.dev/private/service/internal/shared"
)

// RenderOutput calls r.Render and r.WriteContentType on a Renderer
// interface.  This mirrors gin's context.go calling r.Render and
// r.WriteContentType on a render.Render interface — the key bug case
// where an interface method is called across packages.
func RenderOutput(r shared.Renderer, dest string) error {
	ct := r.WriteContentType()
	_ = ct
	return r.Render(dest)
}
