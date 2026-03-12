#include "orderbook/order_book.h"

#include <chrono>
#include <cstdio>
#include <random>

using namespace orderbook;

static double measure(const char* label, int ops, auto fn) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::printf("  %-35s %8d ops  %10.1f us  %8.0f ops/us\n",
                label, ops, us, ops / us);
    return us;
}

int main() {
    std::printf("FlowState Order Book Benchmark\n");
    std::printf("================================================================\n");

    constexpr int N = 1'000'000;
    std::mt19937_64 rng(42);

    // --- Add orders (no crossing) ---
    {
        OrderBook book(0, 200'000, 0.01);
        std::uniform_int_distribution<PriceTicks> bid_dist(90'000, 99'999);
        std::uniform_int_distribution<PriceTicks> ask_dist(100'001, 110'000);
        std::uniform_int_distribution<Quantity> qty_dist(1, 1000);

        measure("add (no crossing)", N, [&] {
            for (int i = 0; i < N; i++) {
                Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
                PriceTicks price = (side == Side::Buy)
                    ? bid_dist(rng) : ask_dist(rng);
                book.add_order(Order{
                    .id = static_cast<OrderId>(i + 1),
                    .side = side,
                    .price = price,
                    .quantity = qty_dist(rng),
                    .timestamp = static_cast<Timestamp>(i),
                });
            }
        });
        std::printf("    -> active orders: %llu, bid levels: %zu, ask levels: %zu\n",
                    static_cast<unsigned long long>(book.stats().active_orders),
                    book.stats().active_bid_levels,
                    book.stats().active_ask_levels);
    }

    // --- Add + cancel (typical HFT pattern) ---
    {
        OrderBook book(0, 200'000, 0.01);
        std::uniform_int_distribution<PriceTicks> price_dist(99'990, 100'010);
        std::uniform_int_distribution<Quantity> qty_dist(1, 100);

        measure("add + cancel (tight spread)", N, [&] {
            for (int i = 0; i < N; i++) {
                auto id = static_cast<OrderId>(i + 1);
                Side side = (i % 2 == 0) ? Side::Buy : Side::Sell;
                book.add_order(Order{
                    .id = id,
                    .side = side,
                    .price = price_dist(rng),
                    .quantity = qty_dist(rng),
                    .timestamp = static_cast<Timestamp>(i),
                });
                // Cancel every other order (simulates quote flickering)
                if (i > 0 && i % 2 == 0) {
                    book.cancel_order(static_cast<OrderId>(i));
                }
            }
        });
    }

    // --- Aggressive matching (every order crosses) ---
    {
        OrderBook book(0, 200'000, 0.01);
        std::uniform_int_distribution<Quantity> qty_dist(1, 100);

        // Seed the book with 10K levels
        for (int i = 0; i < 10'000; i++) {
            book.add_order(Order{
                .id = static_cast<OrderId>(i + 1),
                .side = Side::Sell,
                .price = 100'000 + static_cast<PriceTicks>(i),
                .quantity = 1000,
                .timestamp = static_cast<Timestamp>(i),
            });
        }

        int crossing = N / 10;
        measure("aggressive matching", crossing, [&] {
            for (int i = 0; i < crossing; i++) {
                book.add_order(Order{
                    .id = static_cast<OrderId>(100'000 + i),
                    .side = Side::Buy,
                    .price = 100'005,  // crosses best ask
                    .quantity = qty_dist(rng),
                    .timestamp = static_cast<Timestamp>(100'000 + i),
                });
            }
        });
        std::printf("    -> total fills: %llu\n",
                    static_cast<unsigned long long>(book.stats().total_fills));
    }

    // --- BBO lookup throughput ---
    {
        OrderBook book(0, 200'000, 0.01);
        book.add_order(Order{.id = 1, .side = Side::Buy, .price = 100'000,
                             .quantity = 100, .timestamp = 0});
        book.add_order(Order{.id = 2, .side = Side::Sell, .price = 100'001,
                             .quantity = 100, .timestamp = 0});

        volatile PriceTicks sink = 0;
        measure("BBO lookup", N, [&] {
            for (int i = 0; i < N; i++) {
                auto bbo = book.bbo();
                sink = bbo.bid_price;
            }
        });
        (void)sink;
    }

    return 0;
}
