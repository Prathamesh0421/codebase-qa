; Additive override for TypeScript. tree-sitter-typescript's own tags.scm
; (queries/tags.scm in the wheel) covers signatures, interfaces and abstract
; classes, but has no @reference.call at all, and no plain
; function_declaration / class_declaration / method_definition -- only the
; abstract_* variants. Concretely, ordinary code like this produces nothing
; from the upstream query:
;
;   function greet(name: string) { return sayHello(name); }
;   class Greeter { greet(name: string) { return this.helper(name); } }
;
; These patterns are borrowed from tree-sitter-javascript's tags.scm and
; verified against the actual TypeScript and TSX grammars (both node shapes
; are identical to JS for these constructs) -- see docs/deep-dive.html,
; Phase 2, for the verification transcript. Loaded concatenated with the
; upstream query, so both sets of patterns are active together.

(function_declaration
  name: (identifier) @name) @definition.function

(class_declaration
  name: (type_identifier) @name) @definition.class

(method_definition
  name: (property_identifier) @name) @definition.method

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (member_expression
    property: (property_identifier) @name)) @reference.call
