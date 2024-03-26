"""
The `LazyStruct` table is used for functional-style lazy evaluation.  We set
entries to a function that can produce the correct value; each such function is
invoked the first time it is needed, and aftwerward we use a cached value.

For example:
```
s = LazyStruct()
s.x = () -> 2        # Function that returns 2
s.y = () -> s.x * 3  # Function that returns the result of s.x (=2) * 3
println(s.y)         # Calls s.y to get a result; that in turn calls s.x
println(s.y)         # This time, use the cached result (6)
```
"""
struct LazyStruct
    thunks
    values
    LazyStruct() = new(Dict(), Dict())
    LazyStruct(s :: LazyStruct) = new(copy(s.thunks), copy(s.values))
end


"""
When a property is set on the LazyStruct, we store it as a thunk. The thunk
set is typically a function that returns the value of the property.
"""

function Base.setproperty!(s :: LazyStruct, v :: Symbol, f::Function)
    thunks = getfield(s, :thunks)
    thunks[v] = f
    delete!(getfield(s, :values), v)
end

"""
When a property is accessed, we first check if it is in the values table. If
so, we return the value.  If not, we check if it is in the thunks table.  If
so, we call the thunk to get the value, store it in the values table, and
return it.  If it is not in either table, we return the value of the property
in the struct itself.
"""

function Base.getproperty(s :: LazyStruct, v :: Symbol)
    values = getfield(s, :values)
    thunks = getfield(s, :thunks)
    if haskey(values, v)
        return values[v]
    elseif haskey(thunks, v)
        values[v] = thunks[v]()
        return values[v]
    end
    getfield(s, v)
end

"""
We can also set values directly, without using a thunk.  This is useful for
values that are not computed lazily.
"""

function set(s :: LazyStruct, k :: Symbol, v)
    getfield(s, :values)[k] = v
end
