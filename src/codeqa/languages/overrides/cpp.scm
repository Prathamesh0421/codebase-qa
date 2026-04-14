; Additive override for C++. tree-sitter-cpp's own tags.scm has
; @definition.function, @definition.class, @definition.method but no
; @reference.call at all -- call sites are simply not captured upstream.
; These two patterns cover free-function calls and member calls through
; both `.` and `->`, verified against the actual grammar (field_expression
; is the node type for both `obj.method()` and `obj->method()` -- the
; grammar does not distinguish them structurally). See docs/deep-dive.html,
; Phase 2. Loaded concatenated with the upstream query.

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (field_expression
    field: (field_identifier) @name)) @reference.call
