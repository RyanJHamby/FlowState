#pragma once

#include <cstdint>
#include <limits>

namespace orderbook {

/// Price represented as integer ticks — never floating point.
///
/// Floating-point price comparison (==, <, >) is undefined behavior in
/// practice: IEEE 754 rounding means two prices that should be equal
/// often differ by an ULP. Every production order book uses integer
/// ticks: price_ticks = round(price_double / tick_size). Tick size is
/// set by the exchange (e.g., $0.01 for equities, $0.0001 for FX).
using PriceTicks = int64_t;
using OrderId = uint64_t;
using Quantity = uint64_t;
using Timestamp = int64_t;  // nanoseconds since epoch

constexpr PriceTicks kInvalidPrice = std::numeric_limits<PriceTicks>::min();
constexpr OrderId kInvalidOrderId = 0;

enum class Side : uint8_t {
    Buy = 0,
    Sell = 1,
};

/// Order lifecycle states.
enum class OrderStatus : uint8_t {
    Active = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,
};

/// A single order in the book.
struct Order {
    OrderId id;
    Side side;
    PriceTicks price;
    Quantity quantity;
    Quantity filled{0};
    Timestamp timestamp;  // arrival time, used for FIFO priority
    OrderStatus status{OrderStatus::Active};

    [[nodiscard]] Quantity remaining() const { return quantity - filled; }
    [[nodiscard]] bool is_active() const {
        return status == OrderStatus::Active ||
               status == OrderStatus::PartiallyFilled;
    }
};

/// Execution report — emitted when orders match.
struct Fill {
    OrderId aggressive_id;
    OrderId passive_id;
    PriceTicks price;
    Quantity quantity;
    Timestamp timestamp;
};

/// Best bid and offer snapshot.
struct BBO {
    PriceTicks bid_price{kInvalidPrice};
    Quantity bid_quantity{0};
    PriceTicks ask_price{kInvalidPrice};
    Quantity ask_quantity{0};

    [[nodiscard]] bool has_bid() const { return bid_price != kInvalidPrice; }
    [[nodiscard]] bool has_ask() const { return ask_price != kInvalidPrice; }
    [[nodiscard]] PriceTicks spread() const {
        if (!has_bid() || !has_ask()) return kInvalidPrice;
        return ask_price - bid_price;
    }
};

/// Book-level statistics.
struct BookStats {
    uint64_t total_orders{0};
    uint64_t total_fills{0};
    uint64_t total_cancels{0};
    uint64_t active_orders{0};
    std::size_t active_bid_levels{0};
    std::size_t active_ask_levels{0};
};

}  // namespace orderbook
