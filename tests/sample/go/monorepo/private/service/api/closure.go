package api

// This file reproduces the gin recovery.go bug: goto_definition returns
// empty {} for method calls inside function literals (closures) that are
// not immediately invoked.
//
// In gin's recovery.go, CustomRecoveryWithWriter returns a func(c *Context)
// (a HandlerFunc).  Inside that returned closure, calls like c.Next(),
// c.Error(), c.Abort() all return {} from goto_definition, while the same
// calls in a non-closure function resolve correctly.
//
// The root cause is in the column computation: tree-sitter fails to locate
// the selector_expression node when it is inside a function literal body
// that is assigned, returned, or passed as an argument (but NOT when the
// function literal is immediately invoked with ()).

// ClosureReturned: returns a closure that calls c.Publish.
// c.Publish at line 12 must resolve to controller.go:16.
func ClosureReturned(c *Controller) func() map[string]string {
	return func() map[string]string {
		return c.Publish("returned", 1)
	}
}

// ClosureAssigned: assigns a closure to a variable, then calls it.
// c.Publish at line 20 must resolve to controller.go:16.
func ClosureAssigned(c *Controller) map[string]string {
	f := func() map[string]string {
		return c.Publish("assigned", 1)
	}
	return f()
}

// ClosurePassed: passes a closure as a function argument.
// c.Publish at line 28 must resolve to controller.go:16.
func ClosurePassed(c *Controller) map[string]string {
	return ApplyClosure(func() map[string]string {
		return c.Publish("passed", 1)
	})
}

// ApplyClosure is a helper that invokes a closure.
func ApplyClosure(f func() map[string]string) map[string]string {
	return f()
}

// ClosureImmediateInvoke: immediately invokes a func literal.
// c.Publish at line 39 must resolve to controller.go:16 (control case — this works).
func ClosureImmediateInvoke(c *Controller) {
	func() {
		_ = c.Publish("immediate", 1)
	}()
}

// DirectCall: calls c.Publish directly, no closure (control case — this works).
// c.Publish at line 47 must resolve to controller.go:16.
func DirectCall(c *Controller) map[string]string {
	return c.Publish("direct", 1)
}
