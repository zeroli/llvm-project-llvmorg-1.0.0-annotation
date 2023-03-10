; This test case is used to test opaque type processing, forward references,
; and recursive types.  Oh my.
; 

%SQ1 = type { int }
%ITy = type opaque
%SQ2 = type { %ITy }
%ITy = type int


%CCC = type { \2* }
%BBB = type { \2*, \2 * }
%AAA = type { \2*, {\2*}, [12x{\2*}], {[1x{\2*}]} }

; Test numbered types
type %CCC
type %BBB
%Composite = type { %0, %1 }

; Test simple opaque type resolution...
%intty = type opaque
%intty = type int

; Perform a simple forward reference...
%ty1 = type { %ty2, int }
%ty2 = type float

; Do a recursive type...
%list = type { %list * }
%listp = type { %listp } *

; Do two mutually recursive types...
%TyA = type { %ty2, %TyB * }
%TyB = type { double, %TyA * }

; A complex recursive type...
%Y = type { {%Y*}, %Y* }
%Z = type { { %Z * }, [12x%Z] *, {{{ %Z * }}} }

; More ridiculous test cases...
%A = type [ 123x %A*]
%M = type %M (%M, %M) *
%P = type %P*

; Recursive ptrs
%u = type %v*
%v = type %u*

; Test the parser for unnamed recursive types...
%P1 = type \1 *
%Y1 = type { { \3 * }, \2 * }
%Z1 = type { { \3 * }, [12x\3] *, { { { \5 * } } } }

implementation



