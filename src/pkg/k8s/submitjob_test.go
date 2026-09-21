// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

package k8s

import (
	"context"
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

func testJob(name string) *batchv1.Job {
	return &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "slemify"}}
}

// A deploy that returns while a --no-wait training is still running must
// attach to the running Job, not delete and restart it (an attendee's
// in-flight fine-tune was being killed by the returning deploy).
func TestSubmitJobAttachesToRunningJob(t *testing.T) {
	running := testJob("p-training")
	running.Status.Active = 1
	running.UID = "original"
	cs := fake.NewSimpleClientset(running)
	c := &Client{clientset: cs, namespace: "slemify"}

	name, err := c.SubmitJob(context.Background(), testJob("p-training"))
	if err != nil {
		t.Fatal(err)
	}
	if name != "p-training" {
		t.Fatalf("name = %q", name)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID != "original" {
		t.Fatal("running job was replaced; SubmitJob must attach to it")
	}
}

// A finished job is replaced: re-running a stage explicitly means a fresh run.
func TestSubmitJobReplacesFinishedJob(t *testing.T) {
	done := testJob("p-training")
	done.Status.Succeeded = 1
	done.UID = "original"
	cs := fake.NewSimpleClientset(done)
	c := &Client{clientset: cs, namespace: "slemify"}

	if _, err := c.SubmitJob(context.Background(), testJob("p-training")); err != nil {
		t.Fatal(err)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID == "original" {
		t.Fatal("finished job was not replaced")
	}
}

// A failed job is replaced even if pods are still winding down.
func TestSubmitJobReplacesFailedJob(t *testing.T) {
	failed := testJob("p-training")
	failed.Status.Active = 1
	failed.Status.Failed = 1
	failed.UID = "original"
	cs := fake.NewSimpleClientset(failed)
	c := &Client{clientset: cs, namespace: "slemify"}

	if _, err := c.SubmitJob(context.Background(), testJob("p-training")); err != nil {
		t.Fatal(err)
	}
	got, _ := cs.BatchV1().Jobs("slemify").Get(context.Background(), "p-training", metav1.GetOptions{})
	if got.UID == "original" {
		t.Fatal("failed job was not replaced")
	}
}
