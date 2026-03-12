#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "orderbook/order_book.h"

using namespace orderbook;

// --- Helpers ---

static Order make_order(OrderId id, Side side, PriceTicks price,
                        Quantity qty, Timestamp ts = 0) {
    return Order{.id = id, .side = side, .price = price,
                 .quantity = qty, .timestamp = ts};
}

// --- Types ---

TEST_CASE("Order remaining quantity", "[types]") {
    Order o{.id = 1, .side = Side::Buy, .price = 100, .quantity = 500,
            .timestamp = 0};
    REQUIRE(o.remaining() == 500);
    o.filled = 200;
    REQUIRE(o.remaining() == 300);
    REQUIRE(o.is_active());
    o.status = OrderStatus::Cancelled;
    REQUIRE_FALSE(o.is_active());
}

TEST_CASE("BBO spread calculation", "[types]") {
    BBO bbo;
    REQUIRE_FALSE(bbo.has_bid());
    REQUIRE_FALSE(bbo.has_ask());
    REQUIRE(bbo.spread() == kInvalidPrice);

    bbo.bid_price = 100;
    bbo.ask_price = 102;
    REQUIRE(bbo.has_bid());
    REQUIRE(bbo.has_ask());
    REQUIRE(bbo.spread() == 2);
}

// --- Price Level ---

TEST_CASE("PriceLevel FIFO ordering", "[price_level]") {
    PriceLevel level(100);

    level.add(make_order(1, Side::Buy, 100, 100, 1));
    level.add(make_order(2, Side::Buy, 100, 200, 2));
    level.add(make_order(3, Side::Buy, 100, 50, 3));

    REQUIRE(level.order_count() == 3);
    REQUIRE(level.total_quantity() == 350);

    // Front of queue should be order 1 (earliest)
    REQUIRE(level.orders().front().id == 1);
}

TEST_CASE("PriceLevel cancel", "[price_level]") {
    PriceLevel level(100);
    level.add(make_order(1, Side::Buy, 100, 100));
    level.add(make_order(2, Side::Buy, 100, 200));

    REQUIRE(level.cancel(1));
    REQUIRE(level.order_count() == 1);
    REQUIRE(level.total_quantity() == 200);
    REQUIRE(level.orders().front().id == 2);

    // Cancel nonexistent
    REQUIRE_FALSE(level.cancel(99));
}

TEST_CASE("PriceLevel matching exhausts FIFO", "[price_level]") {
    PriceLevel level(100);
    level.add(make_order(1, Side::Sell, 100, 100, 1));
    level.add(make_order(2, Side::Sell, 100, 200, 2));

    // Aggressive buy for 250 — should fill order 1 fully, order 2 partially
    Order aggressor = make_order(10, Side::Buy, 100, 250, 3);
    std::vector<Fill> fills;
    level.match(aggressor, 3, fills);

    REQUIRE(fills.size() == 2);
    REQUIRE(fills[0].passive_id == 1);
    REQUIRE(fills[0].quantity == 100);
    REQUIRE(fills[1].passive_id == 2);
    REQUIRE(fills[1].quantity == 150);

    REQUIRE(aggressor.remaining() == 0);
    REQUIRE(aggressor.status == OrderStatus::Filled);

    // Order 2 should remain with 50 left
    REQUIRE(level.order_count() == 1);
    REQUIRE(level.orders().front().id == 2);
    REQUIRE(level.orders().front().remaining() == 50);
}

// --- Order Book: basic operations ---

TEST_CASE("Empty book has no BBO", "[book]") {
    OrderBook book(9000, 2000);
    auto bbo = book.bbo();
    REQUIRE_FALSE(bbo.has_bid());
    REQUIRE_FALSE(bbo.has_ask());
}

TEST_CASE("Single bid and ask", "[book]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100));
    book.add_order(make_order(2, Side::Sell, 10005, 200));

    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10000);
    REQUIRE(bbo.bid_quantity == 100);
    REQUIRE(bbo.ask_price == 10005);
    REQUIRE(bbo.ask_quantity == 200);
    REQUIRE(bbo.spread() == 5);
}

TEST_CASE("Multiple bids — best is highest price", "[book]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 9990, 100));
    book.add_order(make_order(2, Side::Buy, 10000, 200));
    book.add_order(make_order(3, Side::Buy, 9995, 150));

    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10000);
    REQUIRE(bbo.bid_quantity == 200);
}

TEST_CASE("Multiple asks — best is lowest price", "[book]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10010, 100));
    book.add_order(make_order(2, Side::Sell, 10005, 200));
    book.add_order(make_order(3, Side::Sell, 10020, 150));

    auto bbo = book.bbo();
    REQUIRE(bbo.ask_price == 10005);
    REQUIRE(bbo.ask_quantity == 200);
}

// --- Crossing: aggressive matching ---

TEST_CASE("Buy order crosses the spread", "[book][matching]") {
    OrderBook book(9000, 2000);

    // Rest a sell at 10005
    book.add_order(make_order(1, Side::Sell, 10005, 100, 1));
    // Buy at 10005 — crosses, should fill immediately
    auto fills = book.add_order(make_order(2, Side::Buy, 10005, 60, 2));

    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].aggressive_id == 2);
    REQUIRE(fills[0].passive_id == 1);
    REQUIRE(fills[0].quantity == 60);
    REQUIRE(fills[0].price == 10005);

    // Remaining 40 on the ask
    auto bbo = book.bbo();
    REQUIRE(bbo.ask_price == 10005);
    REQUIRE(bbo.ask_quantity == 40);
    REQUIRE_FALSE(bbo.has_bid());  // buy fully filled, nothing rests
}

TEST_CASE("Sell order crosses the spread", "[book][matching]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100, 1));
    auto fills = book.add_order(make_order(2, Side::Sell, 10000, 70, 2));

    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].quantity == 70);

    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10000);
    REQUIRE(bbo.bid_quantity == 30);
}

TEST_CASE("Aggressive sweeps multiple levels", "[book][matching]") {
    OrderBook book(9000, 2000);

    // Three ask levels
    book.add_order(make_order(1, Side::Sell, 10005, 100, 1));
    book.add_order(make_order(2, Side::Sell, 10006, 100, 2));
    book.add_order(make_order(3, Side::Sell, 10007, 50, 3));

    // Buy for 220 at 10007 — sweeps through all three levels
    auto fills = book.add_order(make_order(10, Side::Buy, 10007, 220, 4));

    REQUIRE(fills.size() == 3);
    REQUIRE(fills[0].price == 10005);
    REQUIRE(fills[0].quantity == 100);
    REQUIRE(fills[1].price == 10006);
    REQUIRE(fills[1].quantity == 100);
    REQUIRE(fills[2].price == 10007);
    REQUIRE(fills[2].quantity == 20);

    // 30 remaining on ask at 10007
    auto bbo = book.bbo();
    REQUIRE(bbo.ask_price == 10007);
    REQUIRE(bbo.ask_quantity == 30);
}

TEST_CASE("Partial fill leaves remainder in book", "[book][matching]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10005, 50, 1));
    // Buy 100 at 10005 — fills 50 against ask, rests 50 as bid
    auto fills = book.add_order(make_order(2, Side::Buy, 10005, 100, 2));

    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].quantity == 50);

    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10005);
    REQUIRE(bbo.bid_quantity == 50);
    REQUIRE_FALSE(bbo.has_ask());
}

// --- Cancel ---

TEST_CASE("Cancel removes order and updates BBO", "[book][cancel]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100));
    book.add_order(make_order(2, Side::Buy, 9995, 200));

    REQUIRE(book.bbo().bid_price == 10000);

    REQUIRE(book.cancel_order(1));
    REQUIRE(book.bbo().bid_price == 9995);
    REQUIRE(book.bbo().bid_quantity == 200);
}

TEST_CASE("Cancel last order on a side", "[book][cancel]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10005, 100));
    REQUIRE(book.bbo().has_ask());

    REQUIRE(book.cancel_order(1));
    REQUIRE_FALSE(book.bbo().has_ask());
}

TEST_CASE("Cancel nonexistent order", "[book][cancel]") {
    OrderBook book(9000, 2000);
    REQUIRE_FALSE(book.cancel_order(999));
}

// --- Depth ---

TEST_CASE("Bid depth returns levels descending by price", "[book][depth]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100));
    book.add_order(make_order(2, Side::Buy, 9998, 200));
    book.add_order(make_order(3, Side::Buy, 9995, 300));

    auto depth = book.bid_depth(5);
    REQUIRE(depth.size() == 3);
    REQUIRE(depth[0].price == 10000);
    REQUIRE(depth[1].price == 9998);
    REQUIRE(depth[2].price == 9995);
}

TEST_CASE("Ask depth returns levels ascending by price", "[book][depth]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10005, 100));
    book.add_order(make_order(2, Side::Sell, 10010, 200));
    book.add_order(make_order(3, Side::Sell, 10008, 300));

    auto depth = book.ask_depth(5);
    REQUIRE(depth.size() == 3);
    REQUIRE(depth[0].price == 10005);
    REQUIRE(depth[1].price == 10008);
    REQUIRE(depth[2].price == 10010);
}

TEST_CASE("Depth limited to N levels", "[book][depth]") {
    OrderBook book(9000, 2000);

    for (int i = 0; i < 10; i++) {
        book.add_order(make_order(
            static_cast<OrderId>(i + 1), Side::Buy,
            9990 + i, 100));
    }

    auto depth = book.bid_depth(3);
    REQUIRE(depth.size() == 3);
    REQUIRE(depth[0].price == 9999);
}

// --- Statistics ---

TEST_CASE("Stats track orders, fills, cancels", "[book][stats]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100));
    book.add_order(make_order(2, Side::Sell, 10000, 60));

    REQUIRE(book.stats().total_orders == 2);
    REQUIRE(book.stats().total_fills == 1);
    REQUIRE(book.stats().active_orders == 1);  // bid partially filled, rests

    book.cancel_order(1);
    REQUIRE(book.stats().total_cancels == 1);
    REQUIRE(book.stats().active_orders == 0);
}

// --- Tick conversion ---

TEST_CASE("Tick to price conversion", "[book]") {
    OrderBook book(0, 100000, 0.01);
    REQUIRE_THAT(book.tick_to_price(10000),
                 Catch::Matchers::WithinAbs(100.00, 1e-9));
    REQUIRE_THAT(book.tick_to_price(10001),
                 Catch::Matchers::WithinAbs(100.01, 1e-9));
}

// --- Edge cases ---

TEST_CASE("FIFO priority within same price level", "[book][matching]") {
    OrderBook book(9000, 2000);

    // Three sells at same price — order 1 arrived first
    book.add_order(make_order(1, Side::Sell, 10005, 50, 1));
    book.add_order(make_order(2, Side::Sell, 10005, 50, 2));
    book.add_order(make_order(3, Side::Sell, 10005, 50, 3));

    // Buy 80 — should match order 1 (50), then order 2 (30)
    auto fills = book.add_order(make_order(10, Side::Buy, 10005, 80, 4));

    REQUIRE(fills.size() == 2);
    REQUIRE(fills[0].passive_id == 1);
    REQUIRE(fills[0].quantity == 50);
    REQUIRE(fills[1].passive_id == 2);
    REQUIRE(fills[1].quantity == 30);
}

TEST_CASE("Self-trade scenario", "[book][matching]") {
    // Order book does not prevent self-trading at this layer.
    // Self-trade prevention is a gateway/risk concern, not matching engine.
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10000, 100, 1));
    auto fills = book.add_order(make_order(2, Side::Buy, 10000, 100, 2));

    REQUIRE(fills.size() == 1);
    REQUIRE(fills[0].quantity == 100);
}

TEST_CASE("Large order sweeps entire side", "[book][matching]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Sell, 10005, 100, 1));
    book.add_order(make_order(2, Side::Sell, 10010, 100, 2));

    // Buy far above all asks, more quantity than available
    auto fills = book.add_order(make_order(10, Side::Buy, 10500, 500, 3));

    REQUIRE(fills.size() == 2);
    uint64_t total_filled = fills[0].quantity + fills[1].quantity;
    REQUIRE(total_filled == 200);

    // Remaining 300 rests as bid at 10500
    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10500);
    REQUIRE(bbo.bid_quantity == 300);
    REQUIRE_FALSE(bbo.has_ask());
}

TEST_CASE("Quantity aggregation at price level", "[book]") {
    OrderBook book(9000, 2000);

    book.add_order(make_order(1, Side::Buy, 10000, 100));
    book.add_order(make_order(2, Side::Buy, 10000, 200));
    book.add_order(make_order(3, Side::Buy, 10000, 150));

    auto bbo = book.bbo();
    REQUIRE(bbo.bid_price == 10000);
    REQUIRE(bbo.bid_quantity == 450);
}
