#!/bin/sh
set -e

mkdir -p /var/lib/tor /var/log/tor
# Debian's tor package pre-creates /var/lib/tor owned by debian-tor at
# install time, but we run tor as root in this container — it refuses to
# start unless the data directory's owner matches the user running it.
chown -R root:root /var/lib/tor /var/log/tor
chmod 700 /var/lib/tor

tor -f /etc/tor/torrc &

echo "Waiting for Tor to bootstrap..."
until grep -q "Bootstrapped 100%" /var/log/tor/notices.log 2>/dev/null; do
    sleep 1
done
echo "Tor bootstrapped."

exec "$@"
