# User Guide for YAML-DyPDL

In DIDP-YAML, we use YAML to formulate a DyPDL model.
We call our language YAML-DyPDL.
This document describes how to write YAML-DyPDL to model a problem.

To solve a problem using the DyPDL solver, you need to create three files, `domain.yaml`, `problem.yaml`, and `config.yaml`.

## Table of Contents

- [YAML Basics](#yaml-basics)
- [Forbidden Characters](#forbidden-characters)
- [Domain YAML](#domain-yaml)
  - [objects](#objects)
  - [state_variables](#state_variables)
  - [tables](#tables)
  - [dictionaries](#dictionaries)
  - [constraints](#constraints)
  - [base_cases](#base_cases)
  - [reduce](#reduce)
  - [cost_type](#cost_type)
  - [transitions](#transitions)
  - [dual_bounds](#dual_bounds)
- [Problem YAML](#problem-yaml)
  - [object_numbers](#object_numbers)
  - [table_values](#table_values)
  - [dictionary_values](#dictionary_values)
  - [target](#target)

For a config file, see [the solver guide](./solver-guide.md).

## YAML Basics

In YAML, you can use boolean, integer, real, and string values, a list of these values, and a map consisting of key-value pairs.
For example, a menu of a sushi restraunt can be represented by a list of maps as follows:

```yaml
- name: maguro
  price: 3
  description: tuna 
  cooked: false
- name: sake
  price: 3
  description: salmon
  cooked: false
- name: tempura
  price: 5
  description: fried shrimp
  cooked: true
```

In YAML, `-` is used to represent an element of a list.
Data on the left sidef of `:` is a key, and data on the right side of `:` is a value.
Entries in the same list or  must be indented in the same way.
If you are familiar with JSON, the above data is equivalent to the following JSON.

```json
[
    { "name": "maguro", "price": 3, "description": "tuna", "cooked": false },
    { "name": "sake", "price": 3, "description": "salmon", "cooked": false },
    { "name": "tempura", "price": 5, "description": "fried shirmp", "cooked": true }
]
```

In fact, YAML is compatible with JSON, so you can use JSON format inside YAML.
For example, the following is a valid YAML file representing the same data.

```yaml
- { "name": "maguro", "price": 3, "description": "tuna", "cooked": false }
- { name: sake, price: 3, description: "salmon", "cooked": false }
- name: tempura
  price: 5
  description: fried shrimp
  cooked: true
```

In YAML, you do not need to use quotations such as `"` and `'` to represent a string.
However, in the case you want to explicitly represent a digit as a string you should use them.
For example, `5` is parsed as an integer, but `"5"` is parsed as a string.

## Forbidden Characters

When defining a name of something such as objects and state variables, it should not include following characters.

- white space
- tabs
- `(`
- `)`
- `|`

Also, avoid using reserved names used in [expressions](./expression-guide.md)

## Domain YAML

A domain file is used to define problem features that are shared by multiple problem instances.

### objects

`objects` is optional, and the value  is a list of names of object types.
An object type represent a particular type of entity in a problem.
Objects are indexed from `0` to `n-1`, where `n` is defined in the problem file with the [`object_numbers`](#object_numbers) key.
You need to define objects if you use element or set variables.
For example, in a traveling salesperson problem with time windows (TSPTW), a customer is an object.

```yaml
objects:
    - customer
```

### state_variables

`state_variables` is required, and the value is a list of maps describing a state variable.
Each map can have the following keys:

- `name`
- `type`
- `object`
- `preference`

`name` is required, and the value is a string describing the name of the variable.
`type` is required, and the value is either of `set`, `element`, `integer`, or `continuous`.
If `type` is `set` or `element`, defining `object` is required, whose value is the name of an object type defined with `objects` key.

`set` is a set variable, whose value is a set of objects having the specified type.
`element` is an element variable, whose value is an object having the specified type.
`integer` and `continous` are integer and continuous variables.

If `type` is `element`, `integer`, or `continuous`, the key `preference` can be used.
The value for `preference`  is either of `less` or `more`.
Intuitively, with `less`/`more`, if everything else is the same, a state having a smaller/greater value of that variable is better.
Formally, if the values of non-resource variables are the same, a state having equal or better resource variable values must lead to an equal or better solution that has equal or fewer transitions than the other.

#### Example

```yaml
state_variables:
    - name: unvisited
      type: set
      object: customer
    - name: location
      type: element
      object: customer
    - name: time
      type: continuous
      preference: less
```

### tables

`tables` is optional, and the value is a list of maps describing a table of constants.
These values can be accessed from expressions.
Each map can have the following keys:

- `name`
- `type`
- `args`
- `object`
- `default`

`name` is required, and the value is a string describing the name of the table.
`type` is required, and the value is either of `set`, `element`, `integer`, `continuous`, or `bool`.
It represents the type of the constants.

`args` is optional, and the value is a list of names of object types or positive integers.
If it is not defined, the table becomes a constant storing a just one value.
If it is defined, the i-th value defines the size of the i-th dimension.
If it is an object type, the size is the number of the objects
If it is an integer, the size is the integer.

`object` is required for a set table.
It must be an object type or a positive integer.
This value defines the maximum cardinality of sets in the table.
If it is an object type, the maximum cardinality is the number of the objects
If it is an integer, the maximum cardinality is the integer.

`default` is optional, and the value is a list of non-negative integer values if `type` is `set`, an non-negative integer value if `type` is `element`, an integer value if `type` is `integer`, and a real value if `type` is `continuous`.
For a set table default value, the items in the list must be from `0` to `n-1`, where `n` is the maximum cardinality of the set defined in the `object` field.
It represents the default value of the constants, which is used if the value is not defined with the [`table_values`](#tablevalues) key in the problem file.

### dictionaries

`dictionaries` is optional, and the value is a list of maps describing a dictionary of constants.
These values can be accessed from expressions.
Each map can have the following keys:

- `name`
- `type`
- `object`
- `default`

`name` is required, and the value is a string describing the name of the dictionary.
`type` is required, and the value is either of `set`, `element`, `integer`, `continuous`, or `bool`.
It represents the type of the constants.

`object` is required for a set dictionary.
It must be an object type or a positive integer.
This value defines the maximum cardinality of sets in the dictionary.
If it is an object type, the maximum cardinality is the number of the objects
If it is an integer, the maximum cardinality is the integer.

`default` is optional, and the value is a list of non-negative integer values if `type` is `set`, an non-negative integer value if `type` is `element`, an integer value if `type` is `integer`, and a real value if `type` is `continuous`.
For a set dictionary default value, the items in the list must be from `0` to `n-1`, where `n` is the maximum cardinality of the set defined in the `object` field.
It represents the default value of the constants, which is used if the value is not defined with the [`dictionary_values`](#dictionary_values) key in the problem file.

Using dictionaries are not recommended when tables can be used instead.

### constraints

`constraints` is optional, and the value is a list of conditions represented by strings or maps.
It describes conditions that must be satisfied by all states.

If the value is a string, it directly describes a condition.
For the syntax of a condition, please see [the expression guide](./expression-guide.md).
If the value is a map, it should have the following keys.

- `condition`
- `forall`

`condition` is required, and the value is a string describing a condition.
`forall` is required, and the valeue is a map having the following keys.

- `name`
- `object`

`name` is required, and the value is the name of a parameter.
`object` is required, and the value is the name of an object type or a set variable.
This creates a conjunction of conditions over all objects or elements in a set.
In the condition, the object or the element can be accessed by the name of `name`.

#### Example

```yaml
constraints:
  - condition: (<= (+ time (distance location to)) (due_date to))
    forall:
      - name: to
        object: unvisited
```

It can be also defined in a problem file.

### base_cases

`base_cases` is optinal, and the value is a list of lists of conditions.
Alternately,  it can be a list of a map having a key `conditions` whose value is a list of conditions and `cost` whose value is an expression representing the value of the base case.
If `cost` is not given, the value of the base case is 0.
You need to do either defining `base_cases` in a domain file or a problem file.
Each condition is defined in the same way as [`constraints`](#constraints).
One list of conditions correspond to one base case.
If a state satisfies any of base cases, the value of that state is 0, and the recursion of DP stops.

#### Example

```yaml
base_cases:
  - - (is_empty unvisited)
    - (= location 0)
```

The above definition is equivalent to the following.

```yaml
base_cases:
  - conditions:
      - (is_empty unvisited)
      - (= location 0)
    cost: 0
```

### reduce

`reduce` is required, and the value is either of `min` or `max`.
The name `reduce` comes from the fact that we preform a reduce operation to aggregate the results of cost expressions of applocable transitions.
`min`/`max` means that the problem is minimization/maximizatoin.

#### Example

```yaml
reduce: min
```

It can be also defined in a problem file.

### cost_type

`cost_type` is required, and the value is either of `integer` or `continuous`.
If `integer`, integer expressions must be used in cost expressions and dual bounds.
If `continuous`, continuous expressions can be used.

#### Example

```yaml
cost_type: integer
```

### transitions

`transitions` is required, and the value is a list of maps.
Each map has the following keys.

- `name`
- `parameters`
- `preconditions`
- `effect`
- `cost`

`name` is required, and the value is a string describing the name of a transition.

`parameters` is optinal, and the value is a map having the following keys.

- `name`
- `object`

Here, `name` is required, and the value is a string describing the name of a parameter.
`object` is requried, and the value is the name of an object type or a set variable.
Similarly to `forall` in a condition, with `parameters`, for each object or an element in the set variable, one transition is defined.
The object or the element can be accessed in expressions and conditions used in `preconditions`, `effect`, and `cost`.

`preconditions` is optional, and the value is a list of conditions described in the same way as [`constraints`](#constraints).
A transtion is applied only if all preconditions are satisfied by a state.

`effect` is required, and the value is a map where a key is the name of a state variable, and a the value is an expression describing to which value the variable is updated.
If the name of a state variable is not used as a key of the map, that variable is not updated.
For the syntax of an expression, see [the expression guide](./expression-guide.md).

`cost` is required, and the value is a string describing the cost expression.
It is either an integer or a continuous expression depending on the [cost type](#costtype).
In the cost expressoin, in addition to state variables and tables, you can use `cost`, which represnets the cost of the transformed state by the transition.

#### Example

```yaml
transitions:
  - name: visit
    parameters:
      - name: to
        object: unvisited
    effect:
      unvisited: (remove to unvisited)
      location: to
      time: (max (+ time (distance location to)) (ready_time to))
    cost: (+ cost (distance location to))
  - name: return
    preconditions:
      - (is_empty unvisited)
      - (!= location 0)
    effect:
      location: 0
      time: (+ time (distance location 0))
    cost: (+ cost (distance location 0))

```

### dual_bounds

`dual_bounds` is optional, and the value is a list of integer or continuous expressions depending on the [cost type](#costtype).

#### Example

```yaml
dual_bounds:
  - 0
```

It can be also defined in a problem file.

## Problem YAML

A problem file is used to define problem features that are specific to particular problem instances.

### object_numbers

`object_numbers` is optional, and the value is a map describing the numbers of objects.
This must be defined if [`objects`](#objects) is defined in a domain file.
The names of all object types must be included as keys in the map.
The value is a positive integer value describing the number of objects.

#### Example

```yaml
object_numbers:
      customer: 4
```

### target

`target` is required, and the value is a map defining the target state.
The objective of an DIDP model is to compute the value of the target state.
The names of all state variables must be included as keys in the map.
The value is a list of non-negative integer values for a set variable, an non-negative integer value for an element variable, an integer value if for an integer variable, and a real value for a continuous variable.

### Example

```yaml
target:
      unvisited: [ 1, 2, 3 ]
      location: 0
      time: 0
```

### table_values

`table_values` is optional, and the value is a map defining the values of tables.
This must be defined if [`tables`](#tables) is defined in a domain file.
The names of all tables must be included as keys in the map.

If `args` is not defined for a table, the value is a list of non-negative integer values for a set variable, an non-negative integer value for an element variable, an integer value if for an integer variable, and a real value for a continuous variable.
For a set table and a element table, an object used in the value must be from `0` to `n-1` where `n` is the number of objects defined with [`object_numbers`](#objectnumbers) in a problem file.

If `args` is defined, the value is a map.
A key of the map is a list of non-negative integer values representing an object.
The length of the list must be the same as the number of the values of `args`.
For a set table, the items in the each set must be from `0` to `n-1` where `n` is the maximum cardinality of the set defined by the `object` fields.
If you do not include a combination of objects as a key in the map, the default value defined in [`tables`](#tables) is used.
If the default value is not defined, an empty set is used for set tables, and `0` is used for element, integer, and continuous tables, and `false` is used for boolean tables.

#### Example

```yaml
ready_time: { 0: 0, 1: 10, 2: 100, 3: 1000 }
due_date: { 1: 20000, 2: 100, 3: 1000 }
distance: {
            [0, 1]: 10, [0, 2]: 20, [0, 3]: 30,
            [1, 0]: 10, [1, 2]: 30, [1, 3]: 40,
            [2, 0]: 20, [2, 1]: 30, [2, 3]: 50,
            [3, 0]: 30, [3, 1]: 30, [3, 2]: 50
          }
```

In this case, if you use `(due_date 0)` or `(distance 0 0)` in an expression, it returns the default value.

### dictionary_values

`dictionary_values` is optional, and the value is a map defining the values of dictionaries.
This must be defined if [`dictionaries`](#dictionaries) is defined in a domain file.
The names of all dictionaries must be included as keys in the map.

The value is a map.
A key of the map is a list of non-negative integer values representing an object.
The length of the list must be the same as the number of the values of `args`.
For a set dictionary, the items in the each set must be from `0` to `n-1` where `n` is the maximum cardinality of the set defined by the `object` fields.
If you do not include a combination of objects as a key in the map, the default value defined in [`dictionaries`](#dictionaries) is used.
If the default value is not defined, an empty set is used for set dictionaries, `0` is used for element, integer, and continuous dictionaries, and `false` is used for boolean dictionaries.
