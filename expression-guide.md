# User Guide for Expressions in DyPDL

This document describes the syntax of expressions, which are used to describe base cases, constraints, and transitions.

## TIPS
When writing a long expression, you can use multiple lines by placing `>` before a string.
For example,

```yaml
solver: expression_astar
config:
   h: >
      (max (max (ceiling (/ (- (sum time uncompleted) idle-time) cycle-time))
                (- (+ (sum lb2-weight1 uncompleted)
                      (ceiling (sum lb2-weight2 uncompleted)))
                   (if (>= idle-time (/ cycle-time 2)) 1.0 0.0)))
           (- (ceiling (sum lb3-weight uncompleted))
              (if (>= idle-time (/ cycle-time 3)) 1.0 0.0)))
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
- [Numeric Expression](#numeric-expression)
    - [Immediate Value](#immediate-value-1)
    - [Table](#table-2)
    - [sum](#sum)
    - [Variable](#variable-1)
    - [Arithmetic Operations](#arithmetic-operations-1)
    - [Cardinality](#cardinality)
    - [if](#if-2)
- [Condition](#numeric-expression)
    - [Table](#table-3)
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
(max <element expression 1> <element expression 2>)
(min <element expression 1> <element expression 2>)
```

For two element expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), the maximum (`max`), and the minimum (`min`) are defined.

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

## Numeric Expression
A numeric expression can be an integer expression or a continuous expression.
An effect on an integer/continuous variable must be an integer/continuous expression.
Integer immediate values, tables, variables can be used in continuous expressions, but not vice versa.
If you want to use continuous values somewhere in computation, use continuous variables and tables even if the values of them are supposed to be integer.
If continuous immediate values, variables and tables are used, the cost expression is parsed as a continuous expression.
Otherwise, it is parsed as an integer expression.
Even if the cost is integer, if you use a continuous lower bound in the solver, make it contiuous by using continuous immidiate values, tables, and variables.

### Immediate Value

### Table
```
(<table name> <element expression 1>, ..., <element expression k>)
```

It returns a value in table `<table name>` with indices `<element expression 1>` to `<element expression k>`.
The number of element expressions must be the same as `args` of the table.
An integer table can be used in a continuous expression, but not vice versa.

### sum
```
(sum <table name> <element expression 1>|<set expression 1>, ..., <element expression k>|<set expression k>)
```

It returns the sum of values in table `<table name>` with indices specified by the arguments.
It takes the sum over all elements in the cartesian product of the arguments.

For example, suppose that a table named `table1` is 3-dimensional.
`(sum set1 2 set2)` where `set1 = { 0, 1 }` and `set2 = { 3, 4 }` returns the sum of `(table1 0 2 3)`, `(table1 0 2 4)`, `(table1 1 2 3)`, and `(table1 1 2 4)`.

### Variable
```
<variable name>
```

It returns element the value of element variable `<variable name>`.
An integer variable can be used in a continuous expression, but not vice versa.

### Arithmetic Operations

```
(+ <integer expression 1> <integer expression 2>)
(- <integer expression 1> <integer expression 2>)
(* <integer expression 1> <integer expression 2>)
(/ <integer expression 1> <integer expression 2>)
(max <integer expression 1> <integer expression 2>)
(min <integer expression 1> <integer expression 2>)
(abs <integer expression>)
```

For two integer expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), the maximum (`max`), and the minimum (`min`) are defined.
Taking the absolute value of an integer expression (`abs`) is also possible.

```
(+ <continuous expression 1> <continuous expression 2>)
(- <continuous expression 1> <continuous expression 2>)
(* <continuous expression 1> <continuous expression 2>)
(/ <continuous expression 1> <continuous expression 2>)
(max <continuous expression 1> <continuous expression 2>)
(min <continuous expression 1> <continuous expression 2>)
(abs <continuous expression>)
(sqrt <continuous expression>)
(floor <continuous expression>)
(ceiling <continuous expression>)
```

For two integer expressions, addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), the maximum (`max`), and the minimum (`min`) are defined.
Taking the absolute value (`abs`) and the square root (`sqrt`) is also possible.
`floor` returns the maximum integer less than or equal to the value of `<continuous expression>`, and `ceiling` returns the minumum integer greater than or equal to the value of `<continuous expression>`.
However, these functions are continuous expressions, not integer expressions.

### Cardinality
```
|<set expression>|
```

It returns the cardinality of `<set expression>`.

### if
```
(if <condition> <integer expression 1> <integer expression 2>)
(if <condition> <continuous expression 1> <continuous expression 2>)
```

It retunrs `<integer expression 1>`/`<continuous expression 1>` if `<condition>` is true.
Otherwise, it returns `<integer expression 2>`/`<continuous expression 2>`.

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