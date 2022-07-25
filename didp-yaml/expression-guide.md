# User Guide for Expressions in DyPDL

This document describes the syntax of expressions, which are used to describe base cases, constraints, and transitions.

## TIPS
When writing a long expression, you can use multiple lines by placing `>` before a string.
For example,

```yaml
dual_bounds: 
    - >
      (max (max (ceil(/ (- (sum time uncompleted) idle-time) cycle-time))
                (- (+ (sum lb2-weight1 uncompleted)
                      (ceil(sum lb2-weight2 uncompleted)))
                   (if (>= idle-time (/ cycle-time 2.0)) 1 0)))
           (- (ceil(sum lb3-weight uncompleted))
              (if (>= idle-time (/ cycle-time 3.0)) 1 0)))
```


## Table of Contents

- [Element Expression](#element-expression)
    - [Immediate Value](#immediate-value)
    - [Table](#table)
    - [Parameter](#parameter)
    - [Variable](#variable)
    - [Arithmetic Operations](#arithmetic-operations)
    - [if](#if)
- [Set Expression](#set-expression)
    - [Immediate Value](#immediate-value-1)
    - [Table](#table-1)
    - [Variable](#variable-1)
    - [complement](#complement)
    - [union](#union)
    - [intersection](#intersection)
    - [difference](#difference)
    - [add](#add)
    - [remove](#remove)
    - [if](#if-1)
- [Integer Expression](#integer-expression)
    - [Immediate Value](#immediate-value-1)
    - [Table](#table-2)
    - [sum](#sum)
    - [Variable](#variable-1)
    - [Arithmetic Operations](#arithmetic-operations-1)
    - [Rounding](#rounding)
    - [Cardinality](#cardinality)
    - [if](#if-2)
- [Continuous Expression](#continuous-expression)
    - [Immediate Value](#immediate-value-2)
    - [Table](#table-3)
    - [sum](#sum-1)
    - [Variable](#variable-2)
    - [Arithmetic Operations](#arithmetic-operations-2)
    - [Rounding](#rounding-1)
    - [Cardinality](#cardinality-1)
    - [if](#if-3)
- [Condition](#numeric-expression)
    - [Table](#table-4)
    - [Arithmetic Comparison](#arithmetic-comparison)
    - [is_in](#isin)
    - [is_subset](#issubset)
    - [is_empty](#isempty)
    - [not](#not)
    - [and](#and)
    - [or](#or)

## Element Expression
An effect on an element variable must be an element expression.
Also, element expressions are used to access tables.
The value of an element expression must be non-negative and less than the number of the assosiated object.

### Immediate Value
A nonzero integer value is an element expression.

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It returns a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The number of element expressions must be the same as `args` of the table.

### Parameter
```
<parameter name>
```

It returns the value of parameter `<parameter name>`.
Parameter are defined with `forall` in conditions and `parameters` in transitions.

### Variable
```
<variable name>
```

It returns element the value of element variable `<variable name>`.

### Arithmetic Operations

```
(+ <element expression 1> <element expression 2>)
(- <element expression 1> <element expression 2>)
(* <element expression 1> <element expression 2>)
(/ <element expression 1> <element expression 2>)
(% <element expression 1> <element expression 2>)
(max <element expression 1> <element expression 2>)
(min <element expression 1> <element expression 2>)
```

For two element expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), modulus (`%`), the maximum (`max`), and the minimum (`min`) are defined.

### if
```
(if <condition> <element expression 1> <element expression 2>)
```

It retunrs `<element expression 1>` if `<condition>` is true.
Otherwise, it returns `<element expression 2>`.

## Set Expression
An effect on an set variable must be a set expression.

### Immediate Value
```
(<object name> <parameter 1>|<element constant 1>|<element immediate 1> , ..., <parameter k>|<element constant k>|<element immediate k>)
```

It returns a set of objects with type `<object name>` consisting of the argument. 
Each argument is an element expression but restricted to a parameter, an element table having no `args`, and an immediate value.

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It retunrs a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The number of element expressions must be the same as `args` of the table.

### Variable
```
<variable name>
```

It returns element the value of element variable `<variable name>`.

### complement
```
~<set expression>
```

It retunrs a complement set of the value of `<set expression>`.

### union
```
(union <set expression 1> <set expression 2>)
```

It returns the union of `<set expression 1>` and `<set expression 2>`.

### intersection
```
(intersection <set expression 1> <set expression 2>)
```

It returns the intersection of `<set expression 1>` and `<set expression 2>`.

### difference
```
(difference <set expression 1> <set expression 2>)
```

It returns the differene of `<set expression 1>` and `<set expression 2>`, i.e., the intersection of `<set expression 1>` and the complement set of `<set expression 2>`.

### add
```
(add <element expression> <set expression>)
```

It returns the set containing all elements in `<set expression>` in addition to `<element expression>`.

### remove
```
(remove <element expression> <set expression>)
```

It returns the set containing all elements in `<set expression>` except for `<element expression>`.

### if
```
(if <condition> <set expression 1> <set expression 2>)
```

It retunrs `<set expression 1>` if `<condition>` is true.
Otherwise, it returns `<set expression 2>`.

## Integer Expression
An integer expression is a numeric expression using integer values.
An effect on an integer variable must be an integer expression.
If `cost_type` is `integer`, the cost exprssion of each transition and dual bounds must be integer expressions.

### Immediate Value
An integer is an integer expression.

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It returns a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The number of element expressions must be the same as `args` of the table.

### sum
```
(sum <table name> <element expression 1>|<set expression 1>, ..., <element expression k>|<set expression k>)
```

It returns the sum of values in table `<table name>` with indices specified by the arguments.
It takes the sum over all elements in the cartesian product of the arguments.

For example, suppose that a table named `table1` is 3-dimensional.
`(sum table1 set1 2 set2)` where `set1 = { 0, 1 }` and `set2 = { 3, 4 }` returns the sum of `(table1 0 2 3)`, `(table1 0 2 4)`, `(table1 1 2 3)`, and `(table1 1 2 4)`.

### Variable
```
<variable name>
```

It returns the value of integer variable `<variable name>`.

### Arithmetic Operations

```
(+ <integer expression 1> <integer expression 2>)
(- <integer expression 1> <integer expression 2>)
(* <integer expression 1> <integer expression 2>)
(/ <integer expression 1> <integer expression 2>)
(% <integer expression 1> <integer expression 2>)
(max <integer expression 1> <integer expression 2>)
(min <integer expression 1> <integer expression 2>)
(abs <integer expression>)
```

For two integer expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), modulus (`%`), the maximum (`max`), and the minimum (`min`) are defined.
Taking the absolute value of an integer expression (`abs`) is also possible.

### Rounding

```
(ceil <continuous expression>)
(floor <continuous expression>)
(round <continuous expression>)
(trunc <continuous expression>)
```

These expressions convert a continuous expression to an integer expression.

- `ceil` returns the smallest integer that is greater than or equal to the value of the continuous expression.
- `floor` returns the largest integer that does not exceed the value of the continuous expression.
- `round` returns the closest integer.
- `trunc` returns the integer part.

### Cardinality
```
|<set expression>|
```

It returns the cardinality of `<set expression>`.

### if
```
(if <condition> <integer expression 1> <integer expression 2>)
```

It retunrs `<integer expression 1>` if `<condition>` is true.
Otherwise, it returns `<integer expression 2>`.

## Continuous Expression
A continuous expression is a numeric expression using continuous values.
An effect on a continuous variable must be a continuous expression.
If `cost_type` is `continuous`, the cost exprssion of each transition and dual bounds must be continuous expressions.

### Immediate Value
A real value is a continuous expression.

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It returns a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The number of element expressions must be the same as `args` of the table.
An integer table can be used in a continuous expression.

### sum
```
(sum <table name> <element expression 1>|<set expression 1>, ..., <element expression k>|<set expression k>)
```

It returns the sum of values in table `<table name>` with indices specified by the arguments.
It takes the sum over all elements in the cartesian product of the arguments.
An integer table can be used in a continuous expression.

For example, suppose that a table named `table1` is 3-dimensional.
`(sum table1 set1 2 set2)` where `set1 = { 0, 1 }` and `set2 = { 3, 4 }` returns the sum of `(table1 0 2 3)`, `(table1 0 2 4)`, `(table1 1 2 3)`, and `(table1 1 2 4)`.

### Variable
```
<variable name>
```

It returns the value of continuous variable `<variable name>`.
An integer variable can also be used in a continuous expression.

### Arithmetic Operations

```
(+ <continuous expression 1> <continuous expression 2>)
(- <continuous expression 1> <continuous expression 2>)
(* <continuous expression 1> <continuous expression 2>)
(/ <continuous expression 1> <continuous expression 2>)
(% <continuous expression 1> <continuous expression 2>)
(pow <continuous expression 1> <continuous expression 2>)
(log <continuous expression 1> <continuous expression 2>)
(max <continuous expression 1> <continuous expression 2>)
(min <continuous expression 1> <continuous expression 2>)
(abs <continuous expression>)
(sqrt <continuous expression>)
```

For two integer expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), modulus (`%`), power (`pow`), logarithm (`log`),  the maximum (`max`), the minimum (`min`) are defined.
For `pow`, the second argument is an exponent.
For `log`, the second argument is a base.
Taking the absolute value (`abs`) and the square root (`sqrt`) is also possible.

### Rounding

```
(ceil <continuous expression>)
(floor <continuous expression>)
(round <continuous expression>)
(trunc <continuous expression>)
```

These expressions make the fractoinal part to be zero.
However, the returned value is still a continuous expression.

- `ceil` returns the smallest integer that is greater than or equal to the value of the continuous expression.
- `floor` returns the largest integer that does not exceed the value of the continuous expression.
- `round` returns the closest integer.
- `trunc` returns the integer part.

### Cardinality
```
|<set expression>|
```

It returns the cardinality of `<set expression>`.

### if
```
(if <condition> <continuous expression 1> <continuous expression 2>)
```

It retunrs `<continuous expression 1>` if `<condition>` is true.
Otherwise, it returns `<continuous expression 2>`.

## Condition
Conditions are used in state constraints and preconditions.
It returns a boolean value, `true` or `false`.
Also, conditions are used in element, set, and numeric expressions with `if`.

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It returns a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The `type` of the table must be `bool`.
The number of element expressions must be the same as `args` of the table.

### Arithmetic Comparison
```
(= <element expression 1> <element expression 2>)
(!= <element expression 1> <element expression 2>)
(> <element expression 1> <element expression 2>)
(>= <element expression 1> <element expression 2>)
(< <element expression 1> <element expression 2>)
(<= <element expression 1> <element expression 2>)
```

Two element expressions can be compared.

```
(= <numeric expression 1> <numeric expression 2>)
(!= <numeric expression 1> <numeric expression 2>)
(> <numeric expression 1> <numeric expression 2>)
(>= <numeric expression 1> <numeric expression 2>)
(< <numeric expression 1> <numeric expression 2>)
(<= <numeric expression 1> <numeric expression 2>)
```

Two numeric expressions can be compared.
It is possible to compare an integer expression and a continuous expression.

### is_in
```
(is_in <element expression> <set expression>)
```

It checks if the value of `<element expression>` is included in the value of `<set expressoin>`.

### is_subset
```
(is_subset <set expression 1> <set expression 2>)
```

It checks if the value of `<set expressoin 1>` is a subset of `<set expressoin 2>`.

### is_empty
```
(is_empty <set expression>)
```

It checks if the value of `<set expressoin>` is an empty set.

### not
```
(not <condition>)
```

It returns the negation of the value of `<condition>`.

### and
```
(and <condition 1> <condition 2>)
```

It returns the conjunction of the values of `<condition 1>` and `<condition 2>`.

### or
```
(or <condition 1> <condition 2>)
```

It returns the disjunction of the values of `<condition 1>` and `<condition 2>`.