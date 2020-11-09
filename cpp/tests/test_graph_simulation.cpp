#include "epidemiology/migration/graph_simulation.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

class MockNodeFunc
{
public:
    MOCK_METHOD(void, invoke, (double t, double dt, int& i), ());
    void operator()(double t, double dt, int& i)
    {
        invoke(t, dt, i);
    };
};

class MockEdgeFunc
{
public:
    MOCK_METHOD(void, invoke, (double t, double dt, int& e, int& n1, int& n2), ());
    void operator()(double t, double dt, int& e, int& n1, int& n2)
    {
        invoke(t, dt, e, n1, n2);
    };
};

TEST(TestGraphSimulation, simulate)
{
    using testing::_;
    using testing::Eq;

    epi::Graph<int, int> g;
    g.add_node(0);
    g.add_node(1);
    g.add_node(2);
    g.add_node(3);
    g.add_edge(0, 1, 0);
    g.add_edge(1, 2, 1);
    g.add_edge(0, 2, 2);
    g.add_edge(3, 0, 3);

    MockEdgeFunc edge_func;
    MockNodeFunc node_func;

    const auto t0   = 1;
    const auto tmax = 3.0;
    const auto dt   = 1.0;

    testing::ExpectationSet node_func_calls;

    node_func_calls += EXPECT_CALL(node_func, invoke(1, 1, Eq(0))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(1, 1, Eq(1))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(1, 1, Eq(2))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(1, 1, Eq(3))).Times(1);

    EXPECT_CALL(edge_func, invoke(2, 1, Eq(0), Eq(0), Eq(1))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(2, 1, Eq(2), Eq(0), Eq(2))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(2, 1, Eq(1), Eq(1), Eq(2))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(2, 1, Eq(3), Eq(3), Eq(0))).Times(1).After(node_func_calls);

    node_func_calls += EXPECT_CALL(node_func, invoke(2, 1, Eq(0))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(2, 1, Eq(1))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(2, 1, Eq(2))).Times(1);
    node_func_calls += EXPECT_CALL(node_func, invoke(2, 1, Eq(3))).Times(1);

    EXPECT_CALL(edge_func, invoke(3, 1, Eq(0), Eq(0), Eq(1))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(3, 1, Eq(2), Eq(0), Eq(2))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(3, 1, Eq(1), Eq(1), Eq(2))).Times(1).After(node_func_calls);
    EXPECT_CALL(edge_func, invoke(3, 1, Eq(3), Eq(3), Eq(0))).Times(1).After(node_func_calls);

    auto sim = epi::make_graph_sim(
        t0, dt, g,
        [&node_func](auto&& t, auto&& dt_, auto&& n) {
            node_func(t, dt_, n);
        },
        [&edge_func](auto&& t, auto&& dt_, auto&& e, auto&& n1, auto&& n2) {
            edge_func(t, dt_, e, n1, n2);
        });

    sim.advance(tmax);

    EXPECT_NEAR(sim.get_t(), tmax, 1e-15);
}

TEST(TestGraphSimulation, stopsAtTmax)
{
    using testing::_;
    using testing::Eq;

    epi::Graph<int, int> g;
    g.add_node(0);
    g.add_node(1);
    g.add_edge(0, 1, 0);

    const auto t0   = 1.0;
    const auto tmax = 3.123;
    const auto dt   = 0.076;

    auto sim = epi::make_graph_sim(
        t0, dt, g, [](auto&&, auto&&, auto&&) {}, [](auto&&, auto&&, auto&&, auto&&, auto&&) {});

    sim.advance(tmax);

    EXPECT_NEAR(sim.get_t(), tmax, 1e-15);
}

TEST(TestGraphSimulation, persistentChangesDuringSimulation)
{
    epi::Graph<int, int> g;
    g.add_node(6);
    g.add_node(4);
    g.add_node(8);
    g.add_edge(0, 1, 1);
    g.add_edge(0, 2, 2);
    g.add_edge(1, 2, 3);

    auto node_func = [](auto&& /*t*/, auto&& /*dt*/, auto&& n) {
        ++n;
    };
    auto edge_func = [](auto&& /*t*/, auto&& /*dt*/, auto&& e, auto&& /*n1*/, auto&& n2) {
        ++e;
        ++n2;
    };

    auto t0 = 0;
    auto dt = 1;
    auto sim      = epi::make_graph_sim(t0, dt, g, node_func, edge_func);
    int num_steps = 2;
    sim.advance(t0 + num_steps * dt);

    EXPECT_THAT(sim.get_graph().nodes(),
                testing::ElementsAre(6 + num_steps, 4 + num_steps + num_steps, 8 + num_steps + 2 * num_steps));
    std::vector<epi::Edge<int>> v = {{0, 1, 1 + num_steps}, {0, 2, 2 + num_steps}, {1, 2, 3 + num_steps}};
    EXPECT_THAT(sim.get_graph().edges(), testing::ElementsAreArray(v));
}