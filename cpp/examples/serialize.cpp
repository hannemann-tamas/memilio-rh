/**
 * This example demonstrates using the serialization framework
 * and extending it for your own types.
 * See epidemiology/utils/io.h for more information.
 */

#include "epidemiology/utils/io.h"
#include "epidemiology_io/json_serializer.h"

namespace ioex
{
    struct Foo
    {
        std::string s;
        
        //serialize Foo
        //the IOContext knows the format and can handle errors, all it needs is the data
        template<class IOContext>
        void serialize(IOContext& io) const {
            //create an object to receive the data for this class
            auto obj = io.create_object("Foo");
            //add s to the object with a key so it can found when deserializing
            //the framework knows how to handle strings by itself
            obj.add_element("s", s);
        }

        //deserialize Foo
        //the IOContext provides the data
        template<class IOContext>
        static epi::IOResult<Foo> deserialize(IOContext& io) {
            // Retrieve an object for this class.
            //not all formats use the type string, but some (e.g. XML) require it to tag
            //elements that don't have a specific name, e.g. in a list.
            auto obj = io.expect_object("Foo");
            // Retrieve the data element by key. The tag defines the type of the element.
            auto s_rslt = obj.expect_element("s", epi::Tag<std::string>{});
            // The retrieval of elements may fail if the key cannot be found or the element cannot be
            //converted to the right type. use apply to inspect one or more results
            //and create the object if all lookups were succesful. 
            return epi::apply(io, [](auto&& s) { return Foo{s}; }, s_rslt);
        }
    };

    struct Bar 
    {
        int i;
        std::vector<Foo> foos;

        //serialize Bar
        template<class IOContext>
        void serialize(IOContext& io) const {
            auto obj = io.create_object("Bar");
            obj.add_element("i", i);
            //not all data elements have a name of their own, 
            //some are part of a container. Use add_list to add
            //multiple elements of the same type under the same key.  
            //the framework uses Foo::serialize internally to 
            //serialize the Foo objects in the list. 
            obj.add_list("foos", foos.begin(), foos.end());
        }

        //deserialize Bar
        template<class IOContext>
        static epi::IOResult<Bar> deserialize(IOContext& io) {
            auto obj = io.expect_object("Bar");
            //lookup data elements in the same order as they were added by serialize
            //some formats (e.g. binary) don't support random access lookup.
            auto i_rslt = obj.expect_element("i", epi::Tag<int>{});
            auto foos_rslt = obj.expect_list("foos", epi::Tag<Foo>{});
            //the function passed to apply is allowed to do more than just create the object.
            //e.g. it can validate values and return an error
            return epi::apply(io, [](auto&& i, auto&& foos) -> epi::IOResult<Bar> { 
                //use epi::success or epi::failure to return an IOResult
                if (i >= 0) {
                    return epi::success(Bar{i, std::vector<Foo>{foos.begin(), foos.end()}});
                }
                return epi::failure(epi::StatusCode::OutOfRange, "i must be non-negative.");
            }, i_rslt, foos_rslt);
        }
    };
}

epi::IOResult<void> print_json()
{
    ioex::Bar b {42, {{"Hello"}, {"World"}}};

    //Try to turn the Bar object into a json value.
    auto rslt = epi::serialize_json(b);

    //IOResult can be inspected manually e.g. 
    //if (rslt) { do_something(rslt.value()); return success(); }
    //else { return rslt.as_failure(); }
    //For convenience, the BOOST_OUTCOME_TRY macro can be used. 
    //If the operation failed, the error is returned immediately.
    //If the operation was succesful, the result is unpacked and assigned to a new variable.
    //e.g.
    BOOST_OUTCOME_TRY(js, rslt);
    //could also be BOOST_OUTCOME_TRY(js, epi::serialize_json(b)) in one line
    
    //print json (Json::Value) to console
    //could also write to file or do anything else.
    std::cout << js << std::endl;

    //operation succesful, return void
    return epi::success();
}

epi::IOResult<ioex::Bar> read_json()
{
    //create json to deserialize
    //could also be read from file or stream
    Json::Value js;
    js["i"] = 42;
    js["foos"][0]["s"] = "Hello";
    js["foos"][1]["s"] = "World";

    return epi::deserialize_json(js, epi::Tag<ioex::Bar>{});
}

int main() 
{
    std::cout << "Printing ioex::Bar object...\n";
    auto r = print_json();
    if (r) {
        std::cout << "Success.\n";
    }
    else {
        std::cout << "Error: " << r.error().formatted_message() << "\n";
    }

    std::cout << "Deserializing ioex::Bar object... \n";
    auto bar_rslt = read_json();
    if (bar_rslt) {
        std::cout << "Success.\n";
    }
    else {
        std::cout << "Error: " << r.error().formatted_message() << "\n";
    }
}