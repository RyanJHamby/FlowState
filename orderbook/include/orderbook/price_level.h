#pragma once

#include "orderbook/types.h"

#include <deque>

namespace orderbook {

/// A single price level in the book — a FIFO queue of resting orders.
///
/// Orders at the same price are matched in arrival-time order (FIFO),
/// which is the matching priority used by CME, NYSE, and most major
/// exchanges. Some venues use pro-rata; this implementation is FIFO only.
///
/// Uses std::deque rather than std::list for cache locality: orders at
/// a price level are accessed sequentially during matching, and deque
/// stores elements in contiguous chunks (typically 512 bytes) vs list
/// nodes scattered across the heap.
class PriceLevel {
public:
    PriceLevel() = default;
    explicit PriceLevel(PriceTicks price) : price_(price) {}

    /// Add an order to the back of the FIFO queue.
    void add(const Order& order) {
        total_quantity_ += order.remaining();
        orders_.push_back(order);
    }

    /// Remove an order by ID. Returns true if found.
    bool cancel(OrderId id) {
        for (auto it = orders_.begin(); it != orders_.end(); ++it) {
            if (it->id == id) {
                total_quantity_ -= it->remaining();
                orders_.erase(it);
                return true;
            }
        }
        return false;
    }

    /// Match an aggressive order against this level.
    /// Returns fills and reduces resting order quantities.
    /// Fully filled resting orders are removed from the queue.
    void match(Order& aggressor, Timestamp ts, std::vector<Fill>& fills) {
        while (!orders_.empty() && aggressor.remaining() > 0) {
            Order& passive = orders_.front();
            Quantity fill_qty = std::min(aggressor.remaining(), passive.remaining());

            fills.push_back(Fill{
                .aggressive_id = aggressor.id,
                .passive_id = passive.id,
                .price = price_,
                .quantity = fill_qty,
                .timestamp = ts,
            });

            aggressor.filled += fill_qty;
            passive.filled += fill_qty;
            total_quantity_ -= fill_qty;

            if (passive.remaining() == 0) {
                passive.status = OrderStatus::Filled;
                orders_.pop_front();
            } else {
                passive.status = OrderStatus::PartiallyFilled;
            }
        }

        if (aggressor.remaining() == 0) {
            aggressor.status = OrderStatus::Filled;
        } else {
            aggressor.status = OrderStatus::PartiallyFilled;
        }
    }

    [[nodiscard]] PriceTicks price() const { return price_; }
    [[nodiscard]] Quantity total_quantity() const { return total_quantity_; }
    [[nodiscard]] size_t order_count() const { return orders_.size(); }
    [[nodiscard]] bool empty() const { return orders_.empty(); }

    [[nodiscard]] const std::deque<Order>& orders() const { return orders_; }

private:
    PriceTicks price_{kInvalidPrice};
    Quantity total_quantity_{0};
    std::deque<Order> orders_;
};

}  // namespace orderbook
