#![no_std]
#![no_main]

use core::cell::UnsafeCell;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_println::println;
use fw_esp32c5::algo::{
    EPOCH_FEATURES, EPOCH_LABELS, LABEL_SCHEDULE, MemPool, PREDICT_STEPS, TOTAL_STEPS,
};

esp_bootloader_esp_idf::esp_app_desc!();

struct StaticPool<const K: usize>(UnsafeCell<MemPool<K>>);
unsafe impl<const K: usize> Sync for StaticPool<K> {}

static POOL_EPOCH: StaticPool<10> = StaticPool(UnsafeCell::new(MemPool::new(0)));
static POOL_DUP: StaticPool<10> = StaticPool(UnsafeCell::new(MemPool::new(0)));
static POOL_FULL: StaticPool<4> = StaticPool(UnsafeCell::new(MemPool::new(0)));

fn delay() {
    for _ in 0..100000 {
        core::hint::spin_loop();
    }
}

fn run_epoch() {
    println!("R1 START");
    delay();
    let pool = unsafe { &mut *POOL_EPOCH.0.get() };
    *pool = MemPool::new(0);
    let mut learned = 0u32;

    for step in 0..TOTAL_STEPS {
        if step < PREDICT_STEPS {
            let z = EPOCH_FEATURES[step];
            println!("S{} PRED", step);
            delay();
            let _ = pool.push(z);
        }
        if let Some(lid) = LABEL_SCHEDULE[step] {
            let lab = EPOCH_LABELS[lid as usize];
            let ok = pool.on_label(lid, lab).is_ok();
            println!("S{} LAB {} {}", step, lid, ok);
            delay();
        }
        while let Some(c) = pool.pop_ready() {
            learned += 1;
            println!("S{} LEARN {}", step, c.id);
            delay();
            pool.promote_to_replay(c);
        }
    }
    println!("R1 END L={} M={} miss={} drop={}", learned, pool.matched, pool.miss_labels, pool.drop_total());
    delay();
}

fn run_dup() {
    println!("R2 START");
    delay();
    let pool = unsafe { &mut *POOL_DUP.0.get() };
    *pool = MemPool::new(0);
    let z = EPOCH_FEATURES[0];
    let lab = EPOCH_LABELS[0];
    let _ = pool.push(z);
    let _ = pool.on_label(0, lab);
    if let Some(c) = pool.pop_ready() {
        pool.promote_to_replay(c);
    }
    let ok = pool.on_label(0, lab).is_ok();
    println!("R2 END L={} ok={}", pool.learned_first, ok);
    delay();
}

fn run_full() {
    println!("R3 START");
    delay();
    let pool = unsafe { &mut *POOL_FULL.0.get() };
    *pool = MemPool::new(0);
    for i in 0..5u32 {
        let z = EPOCH_FEATURES[i as usize];
        let _ = pool.push(z);
    }
    for lid in [0u32, 1, 2, 3] {
        let lab = EPOCH_LABELS[lid as usize];
        let _ = pool.on_label(lid, lab);
    }
    let mut learned = 0u32;
    while let Some(c) = pool.pop_ready() {
        learned += 1;
        pool.promote_to_replay(c);
    }
    println!("R3 L={}", learned);
    delay();
    for (k, (id, reason)) in pool.drop_journal().iter().enumerate() {
        println!("DROP {} {} {:?}", k, id, reason);
        delay();
    }
    println!("R3 END miss={} drop={}", pool.miss_labels, pool.drop_total());
    delay();
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);
    println!("FW START");
    delay();
    run_epoch();
    run_dup();
    run_full();
    println!("FW END");
    loop {}
}
