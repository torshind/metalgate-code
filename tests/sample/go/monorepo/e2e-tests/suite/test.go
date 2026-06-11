package suite

import (
	"go.example.dev/private/service/api"
)

// Runner drives end-to-end tests.
type Runner struct {
	ctrl *api.Controller
}

// NewRunner creates a test runner with a controller.
func NewRunner() *Runner {
	return &Runner{ctrl: api.NewController()}
}

// RunTest exercises the controller.
func (r *Runner) RunTest() (string, error) {
	ctx := r.ctrl.Publish("test", 42)
	return r.ctrl.Lookup(ctx, "test")
}
